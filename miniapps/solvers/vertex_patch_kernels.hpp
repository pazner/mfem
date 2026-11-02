// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE and NOTICE for details.

// Internal helpers included inside mfem::vertex_patch's anonymous namespace.
// All numerical kernels use MFEM dispatch, including the CPU reference path.
template <typename T>
void CopyArray(const std::vector<T> &src, Array<T> &dst)
{
   dst.SetSize(src.size()); std::copy(src.begin(),src.end(),dst.HostWrite());
}

MFEM_HOST_DEVICE inline void AtomicMinIndex(int &target, int value)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   atomicMin(&target,value);
#else
#ifdef MFEM_USE_OPENMP
   #pragma omp critical
#endif
   { if (value < target) { target = value; } }
#endif
}

// Conforming scalar H1 restriction has exactly one unit entry per row.
// Apply it as an index gather/scatter, without a sparse BLAS call.
inline void BuildTrueDofMap(ParFiniteElementSpace &fes, Array<int> &map)
{
   const SparseMatrix &r = *fes.GetRestrictionMatrix();
   const int n = fes.GetTrueVSize();
   MFEM_VERIFY(r.NumNonZeroElems() == n, "Expected conforming H1 restriction");
   map.SetSize(n); const int *J = r.ReadJ(); int *M = map.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { M[i] = J[i]; });
}
inline void RestrictTrueDofs(const Array<int> &map, const Vector &x,
                             Vector &y, bool transpose)
{
   y.UseDevice(true);
   if (transpose) { y = 0.; }
   const int *M = map.Read(); const real_t *X = x.Read();
   real_t *Y = transpose ? y.ReadWrite() : y.Write();
   mfem::forall(map.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      if (transpose) { Y[M[i]] = X[i]; }
      else { Y[i] = X[M[i]]; }
   });
}

// Symbolic insertion is separated from numerical accumulation. Structural
// zeros remain in the CSR graph, including the prescribed ICC fill.
class SparseAccumulator
{
   int height, width;
public:
   std::unique_ptr<SparseMatrix> matrix;
   SparseAccumulator(int height, int width)
      : height(height), width(width) { }
   void Assemble(const Vector &values, const Array<int> &row,
                 const Array<int> &col,
                 const Array<int> &source, const Vector &weight)
   {
      const int *R = row.Read(), *C = col.Read(), *S = source.Read();
#ifdef MFEM_USE_CUDA
      if (Device::Allows(Backend::CUDA_MASK))
      {
         Array<unsigned long long> keys(row.Size()+1);
         auto *K = keys.Write(); const int w = width, h = height;
         const int count = row.Size();
         mfem::forall(count+1, [=] MFEM_HOST_DEVICE (int k)
         { K[k] = k == count || R[k] < 0 ? ~0ULL : static_cast<unsigned long long>(R[k])*w+C[k]; });
         thrust::sort(thrust::device,K,K+count+1);
         const int nnz = thrust::unique(thrust::device,K,K+count+1)-K-1;
         int *ip = new int[h+1]; ip[h] = nnz;
         matrix.reset(new SparseMatrix(ip,new int[nnz],new real_t[nnz],h,w,true,true,
                                       true));
         int *I = matrix->WriteI(), *J = matrix->WriteJ();
         mfem::forall(nnz, [=] MFEM_HOST_DEVICE (int k) { J[k] = K[k]%w; });
         mfem::forall(h+1, [=] MFEM_HOST_DEVICE (int r)
         {
            const auto key = static_cast<unsigned long long>(r)*w;
            int lo = 0, hi = nnz;
            while (lo < hi) { const int m = (lo+hi)/2; if (K[m] < key) { lo=m+1; } else { hi=m; } }
            I[r] = lo;
         });
      }
      else
#endif
      {
         matrix.reset(new SparseMatrix(height,width));
         const int *rh = row.HostRead(), *ch = col.HostRead();
         for (int k = 0; k < row.Size(); ++k) { if (rh[k] >= 0) { matrix->Add(rh[k],ch[k],0.); } }
         matrix->Finalize(0); matrix->SortColumnIndices();
      }
      R = row.Read(); C = col.Read(); S = source.Read();
      const int *I = matrix->ReadI(), *J = matrix->ReadJ();
      const real_t *V = values.Read(), *W = weight.Read();
      real_t *A = matrix->WriteData();
      mfem::forall(matrix->NumNonZeroElems(), [=] MFEM_HOST_DEVICE (int k) { A[k] = 0.; });
      mfem::forall(row.Size(), [=] MFEM_HOST_DEVICE (int k)
      {
         if (R[k] < 0) { return; }
         int lo = I[R[k]], hi = I[R[k]+1];
         while (lo < hi) { const int m = (lo+hi)/2; if (J[m] < C[k]) { lo=m+1; } else { hi=m; } }
         AtomicAdd(A[lo], W[k]*V[S[k]]);
      });
   }
};

// Eight adjacent lanes cooperate on one sparse row in the level schedule.
// The subgroup mask also handles a final, partially populated CUDA block.
MFEM_HOST_DEVICE inline real_t PatchRowSum(real_t value)
{
#if defined(__CUDA_ARCH__)
   const unsigned mask = 0xffu << (8*(MFEM_THREAD_ID(z)%4));
   for (int offset = 4; offset > 0; offset /= 2)
   { value += __shfl_down_sync(mask,value,offset,8); }
#elif defined(__HIP_DEVICE_COMPILE__)
   for (int offset = 4; offset > 0; offset /= 2)
   { value += __shfl_down(value,offset,8); }
#endif
   return value;
}

// A graph, permutation, and dependency schedule are shared by all patches
// with the same structure. Numerical factors and residuals remain per patch.
class PatchFactors
{
public:
   struct Layout
   {
      int n, nnz, i, j, ti, tj, levels, rows, nlevels, source, diagonal;
   };
   struct Instance { int layout, row, factor, source; };
   struct ScheduledRow { int patch, row; };
   struct Schedule
   {
      int begin = 0, end = 0;
      Array<int> offsets;
      Array<ScheduledRow> rows;
   };
private:
   std::vector<Layout> layouts;
   std::vector<Instance> instances;
   std::vector<int> si, sj, sti, stj, sr, sl, ls, ds, permutation;
   Array<Layout> device_layouts;
   Array<Instance> device_instances;
   Array<int> offsets, columns, transpose_offsets, transpose_entries;
   Array<int> level_offsets, level_rows, lower_source, diagonal_source, gather;
   Vector factors, diagonal;
   mutable Vector work;
   std::vector<int> indices;
   std::vector<long long> vertices;
   std::map<std::vector<int>, int> patterns;
   std::vector<Vector> chunks;
   int source_size = 0, factor_size = 0, max_patch_size = 0;
   Schedule local_schedule, remote_schedule;
   template <typename T>
   static void Copy(const std::vector<T> &src, Array<T> &dst)
   { dst.SetSize(src.size()); std::copy(src.begin(), src.end(), dst.HostWrite()); }
public:
   int AddPattern(const SparseMatrix &matrix, const Array<int> &order)
   {
      const int n = order.Size();
      const int *ai = matrix.HostReadI(), *aj = matrix.HostReadJ();
      const int *o = order.HostRead();
      std::vector<int> key(ai,ai+n+1);
      key.insert(key.end(),aj,aj+matrix.NumNonZeroElems());
      key.insert(key.end(),o,o+n);
      auto found = patterns.find(key);
      if (found != patterns.end()) { return found->second; }
      const int id = layouts.size();
      patterns.emplace(std::move(key),id);
      Layout t{n,0,int(si.size()),int(sj.size()),int(sti.size()),int(stj.size()),
               int(sl.size()),int(sr.size()),0,int(ls.size()),int(ds.size())};
      std::vector<int> inverse(n), levels(n), ii{0}, jj, sources;
      for (int i = 0; i < n; ++i) { inverse[o[i]] = i; }
      permutation.insert(permutation.end(),o,o+n);
      for (int i = 0; i < n; ++i)
      {
         std::vector<std::pair<int,int>> row;
         int diagonal_entry = -1;
         for (int k = ai[o[i]]; k < ai[o[i]+1]; ++k)
         {
            const int j = inverse[aj[k]];
            if (j < i) { row.emplace_back(j,k); }
            if (j == i) { diagonal_entry = k; }
         }
         MFEM_VERIFY(diagonal_entry >= 0,"Missing patch diagonal");
         ds.push_back(diagonal_entry);
         std::sort(row.begin(),row.end());
         for (const auto &entry : row)
         {
            jj.push_back(entry.first); sources.push_back(entry.second);
            levels[i] = std::max(levels[i],levels[entry.first]+1);
         }
         ii.push_back(jj.size());
         t.nlevels = std::max(t.nlevels,levels[i]+1);
      }
      t.nnz = jj.size();
      si.insert(si.end(),ii.begin(),ii.end());
      sj.insert(sj.end(),jj.begin(),jj.end());
      ls.insert(ls.end(),sources.begin(),sources.end());
      // Transpose entries point into the lower factor. Their row can be
      // recovered from a second packed column index, avoiding indirect searches.
      std::vector<int> ti(n+1), tj(jj.size()), tr(jj.size());
      for (int j : jj) { ++ti[j+1]; }
      for (int i = 0; i < n; ++i) { ti[i+1] += ti[i]; }
      auto next = ti;
      for (int i = 0; i < n; ++i) for (int k = ii[i]; k < ii[i+1]; ++k)
         { int a = next[jj[k]]++; tj[a] = k; tr[a] = i; }
      sti.insert(sti.end(),ti.begin(),ti.end());
      stj.insert(stj.end(),tj.begin(),tj.end());
      stj.insert(stj.end(),tr.begin(),tr.end());
      std::vector<std::vector<int>> rows(t.nlevels);
      for (int i = 0; i < n; ++i) { rows[levels[i]].push_back(i); }
      sl.push_back(0);
      int count = 0;
      for (const auto &row : rows)
      { sr.insert(sr.end(),row.begin(),row.end()); count += row.size(); sl.push_back(count); }
      layouts.push_back(t);
      return id;
   }
   int AddInstance(int layout, const Array<int> &map, int source,
                   long long vertex = -1)
   {
      const auto &t = layouts[layout]; const int id = instances.size();
      MFEM_VERIFY(map.Size() == t.n,"Incorrect patch map size");
      instances.push_back({layout,int(indices.size()),factor_size,source});
      const int *m = map.HostRead();
      for (int i = 0; i < t.n; ++i) { indices.push_back(m[permutation[t.diagonal+i]]); }
      vertices.push_back(vertex < 0 ? id : vertex);
      factor_size += t.nnz; max_patch_size = std::max(max_patch_size,t.n);
      return id;
   }
   // Convenience interface for independent matrices (also used by the tests).
   int Add(const SparseMatrix &matrix, const Array<int> &order,
           const Array<int> &map, long long vertex = -1)
   {
      const int id = AddInstance(AddPattern(matrix,order),map,source_size,vertex);
      const int count = matrix.NumNonZeroElems(); source_size += count;
      chunks.emplace_back(count); chunks.back().UseDevice(true);
      const real_t *A = matrix.ReadData(); real_t *B = chunks.back().Write();
      mfem::forall(count,[=] MFEM_HOST_DEVICE (int k) { B[k] = A[k]; });
      return id;
   }
   int Nonzeros() const { return factor_size+std::max(gather.Size(),int(indices.size())); }
   int NumPatterns() const { return std::max(device_layouts.Size(),int(layouts.size())); }
   void Factor(real_t shift, int rank = 0)
   {
      Vector numerical(source_size); numerical.UseDevice(true);
      real_t *A = numerical.Write(); int start = 0;
      for (const auto &chunk : chunks)
      {
         const real_t *B = chunk.Read(); const int base = start;
         mfem::forall(chunk.Size(),[=] MFEM_HOST_DEVICE (int i) { A[base+i] = B[i]; });
         start += chunk.Size();
      }
      Factor(numerical,shift,rank); chunks.clear();
   }
   void Factor(const Vector &numerical, real_t shift, int rank = 0)
   {
      Copy(layouts,device_layouts); Copy(instances,device_instances);
      Copy(si,offsets); Copy(sj,columns); Copy(sti,transpose_offsets);
      Copy(stj,transpose_entries); Copy(sl,level_offsets); Copy(sr,level_rows);
      Copy(ls,lower_source); Copy(ds,diagonal_source); Copy(indices,gather);
      factors.SetSize(factor_size); diagonal.SetSize(indices.size());
      work.SetSize(indices.size());
      factors.UseDevice(true); diagonal.UseDevice(true); work.UseDevice(true);
      const auto *T = device_layouts.Read(); const auto *P = device_instances.Read();
      const int *I = offsets.Read(), *J = columns.Read(), *L = level_offsets.Read();
      const int *R = level_rows.Read(), *S = lower_source.Read(),
                 *DS = diagonal_source.Read();
      const real_t *A = numerical.Read();
      real_t *F = factors.Write(), *D = diagonal.Write();
      Array<int> failure(instances.size()), pivot(instances.size()); failure = 0;
      pivot = -1;
      int *bad = failure.ReadWrite(), *where = pivot.ReadWrite();
      mfem::forall_2D(instances.size(),128,1,[=] MFEM_HOST_DEVICE (int p)
      {
         const auto b = P[p]; const auto t = T[b.layout];
         for (int l = 0; l < t.nlevels; ++l)
         {
            MFEM_FOREACH_THREAD(a,x,L[t.levels+l+1]-L[t.levels+l])
            {
               const int i = R[t.rows+L[t.levels+l]+a];
               real_t d = A[b.source+DS[t.diagonal+i]]; d += shift*fabs(d);
               for (int k = I[t.i+i]; k < I[t.i+i+1]; ++k)
               {
                  const int j = J[t.j+k]; real_t v = A[b.source+S[t.source+k]];
                  int a = I[t.i+i], c = I[t.i+j];
                  while (a < k && c < I[t.i+j+1])
                  {
                     if (J[t.j+a] < J[t.j+c]) { ++a; }
                     else if (J[t.j+c] < J[t.j+a]) { ++c; }
                     else { v -= F[b.factor+a++]*F[b.factor+c++]; }
                  }
                  v /= D[b.row+j]; F[b.factor+k] = v; d -= v*v;
               }
               if (!(d > 0.) || !std::isfinite(d))
               { if (AtomicAdd(bad[p],1) == 0) { where[p] = i; } d = 1.; }
               D[b.row+i] = sqrt(d);
            }
            MFEM_SYNC_THREAD;
         }
      });
      const int *failed = failure.HostRead(), *location = pivot.HostRead();
      for (int p = 0; p < int(instances.size()); ++p)
      {
         MFEM_VERIFY(!failed[p],"ICC nonpositive pivot " <<
                     permutation[layouts[instances[p].layout].diagonal+location[p]] <<
                     " in vertex patch " << vertices[p] << " on rank " << rank);
      }
      // Assembly-only maps are no longer needed after numerical factorization.
      lower_source.DeleteAll(); diagonal_source.DeleteAll();
      patterns.clear(); std::vector<int>().swap(indices);
      for (auto *v : {&si,&sj,&sti,&stj,&sl,&sr}) { std::vector<int>().swap(*v); }
      std::vector<Layout>().swap(layouts); std::vector<Instance>().swap(instances);
      std::vector<int>().swap(ls); std::vector<int>().swap(ds);
      std::vector<int>().swap(permutation); std::vector<long long>().swap(vertices);
   }
   void Prepare(const Array<int> &order, int split)
   {
      const auto *layout = device_layouts.HostRead();
      const auto *instance = device_instances.HostRead();
      const int *sl = level_offsets.HostRead(), *sr = level_rows.HostRead();
      auto build = [&](Schedule &s, int begin, int end)
      {
         s.begin = begin; s.end = end;
         std::vector<std::vector<ScheduledRow>> levels;
         const int *o = order.HostRead();
         for (int k = begin; k < end; ++k)
         {
            const int p = o[k]; const auto &t = layout[instance[p].layout];
            levels.resize(std::max(int(levels.size()),t.nlevels));
            for (int l = 0; l < t.nlevels; ++l)
               for (int j = sl[t.levels+l]; j < sl[t.levels+l+1]; ++j)
               { levels[l].push_back({p,sr[t.rows+j]}); }
         }
         std::vector<int> ends{0}; std::vector<ScheduledRow> rows;
         for (const auto &level : levels)
         { rows.insert(rows.end(),level.begin(),level.end()); ends.push_back(rows.size()); }
         Copy(ends,s.offsets); Copy(rows,s.rows);
      };
      build(local_schedule,0,split); build(remote_schedule,split,order.Size());
   }
   void ApplyLevels(const Vector &x, Vector &y, bool remote) const
   {
      const auto &s = remote ? remote_schedule : local_schedule;
      const auto *Q = s.rows.Read(); const int *ends = s.offsets.HostRead();
      const auto *T = device_layouts.Read(); const auto *P = device_instances.Read();
      const int *I = offsets.Read(), *J = columns.Read(), *G = gather.Read();
      const int *TI = transpose_offsets.Read(), *TJ = transpose_entries.Read();
      const real_t *F = factors.Read(), *D = diagonal.Read(), *X = x.Read();
      real_t *W = work.ReadWrite(), *Y = y.ReadWrite();
      for (int l = 0; l+1 < s.offsets.Size(); ++l)
      {
         const int start = ends[l];
         mfem::forall_2D_batch(ends[l+1]-start,8,1,16,[=] MFEM_HOST_DEVICE (int k)
         {
            const auto q = Q[start+k]; const auto b = P[q.patch];
            const auto t = T[b.layout];
            const int i = q.row; real_t sum = 0.;
            for (int j = I[t.i+i]+MFEM_THREAD_ID(x); j < I[t.i+i+1];
                 j += MFEM_THREAD_SIZE(x))
            { sum += F[b.factor+j]*W[b.row+J[t.j+j]]; }
            sum = PatchRowSum(sum);
            if (MFEM_THREAD_ID(x) == 0) { W[b.row+i] = (X[G[b.row+i]]-sum)/D[b.row+i]; }
         });
      }
      for (int l = s.offsets.Size()-2; l >= 0; --l)
      {
         const int start = ends[l];
         mfem::forall_2D_batch(ends[l+1]-start,8,1,16,[=] MFEM_HOST_DEVICE (int k)
         {
            const auto q = Q[start+k]; const auto b = P[q.patch];
            const auto t = T[b.layout];
            const int i = q.row; real_t sum = 0.;
            for (int j = TI[t.ti+i]+MFEM_THREAD_ID(x); j < TI[t.ti+i+1];
                 j += MFEM_THREAD_SIZE(x))
            { sum += F[b.factor+TJ[t.tj+j]]*W[b.row+TJ[t.tj+t.nnz+j]]; }
            sum = PatchRowSum(sum);
            if (MFEM_THREAD_ID(x) == 0)
            { const real_t v = (W[b.row+i]-sum)/D[b.row+i]; W[b.row+i] = v; AtomicAdd(Y[G[b.row+i]],v); }
         });
      }
   }
   template <int SHARED>
   void ApplyBlocks(const Vector &x, Vector &y, const Array<int> &order,
                    int begin, int end) const
   {
      const auto *T = device_layouts.Read(); const auto *P = device_instances.Read();
      const int *I = offsets.Read(), *J = columns.Read(), *G = gather.Read();
      const int *L = level_offsets.Read(), *R = level_rows.Read(), *O = order.Read();
      const int *TI = transpose_offsets.Read(), *TJ = transpose_entries.Read();
      const real_t *F = factors.Read(), *D = diagonal.Read(), *X = x.Read();
      real_t *global = work.ReadWrite(), *Y = y.ReadWrite();
      mfem::forall_2D(end-begin,128,1,[=] MFEM_HOST_DEVICE (int k)
      {
         const auto b = P[O[begin+k]]; const auto t = T[b.layout];
         MFEM_SHARED real_t scratch[SHARED ? SHARED : 1];
         real_t *W = SHARED ? scratch : global+b.row;
         for (int l = 0; l < t.nlevels; ++l)
         {
            MFEM_FOREACH_THREAD(a,x,L[t.levels+l+1]-L[t.levels+l])
            {
               const int i = R[t.rows+L[t.levels+l]+a]; real_t v = X[G[b.row+i]];
               for (int j = I[t.i+i]; j < I[t.i+i+1]; ++j) { v -= F[b.factor+j]*W[J[t.j+j]]; }
               W[i] = v/D[b.row+i];
            }
            MFEM_SYNC_THREAD;
         }
         for (int l = t.nlevels-1; l >= 0; --l)
         {
            MFEM_FOREACH_THREAD(a,x,L[t.levels+l+1]-L[t.levels+l])
            {
               const int i = R[t.rows+L[t.levels+l]+a]; real_t v = W[i];
               for (int j = TI[t.ti+i]; j < TI[t.ti+i+1]; ++j)
               { v -= F[b.factor+TJ[t.tj+j]]*W[TJ[t.tj+t.nnz+j]]; }
               W[i] = v/D[b.row+i]; AtomicAdd(Y[G[b.row+i]],W[i]);
            }
            MFEM_SYNC_THREAD;
         }
      });
   }
   void Apply(const Vector &x, Vector &y, const Array<int> &order, int begin,
              int end) const
   {
      if (begin == end) { return; }
      const bool remote = !(local_schedule.begin == begin &&
                            local_schedule.end == end);
      const Schedule *s = remote ? &remote_schedule : &local_schedule;
      // Amortize level-kernel launches over a large batch; small batches and
      // deep dependency graphs use one block per patch instead.
      if (Device::Allows(Backend::DEVICE_MASK) && s->begin == begin &&
          s->end == end &&
          end-begin >= 2048 && s->rows.Size() >= 65536 && s->offsets.Size() <= 129)
      { ApplyLevels(x,y,remote); }
      else if (max_patch_size <= 512) { ApplyBlocks<512>(x,y,order,begin,end); }
      else if (max_patch_size <= 1024) { ApplyBlocks<1024>(x,y,order,begin,end); }
      else { ApplyBlocks<0>(x,y,order,begin,end); }
   }
   long long StorageBytes() const
   {
      return sizeof(real_t)*(static_cast<long long>(factors.Size())+diagonal.Size()
                             +work.Size())+
             sizeof(int)*(static_cast<long long>(offsets.Size())+columns.Size()
                          +transpose_offsets.Size()+
                          transpose_entries.Size()+level_offsets.Size()+level_rows.Size()+gather.Size()+
                          local_schedule.offsets.Size()+remote_schedule.offsets.Size())+
             sizeof(Layout)*device_layouts.Size()+sizeof(Instance)*device_instances.Size()+
             sizeof(ScheduledRow)*(static_cast<long long>(local_schedule.rows.Size())
                                   +remote_schedule.rows.Size());
   }
};

// Q1-to-Qp interpolation directly on true DOFs. Each high-order node depends
// on only the cell vertices; the transpose uses exactly the same selection mask.
class LinearCoarseTransfer : public Operator
{
   ParFiniteElementSpace &coarse, &fine;
   Array<int> cm, fm, mask, true_dofs;
   Vector nodes;
   mutable Vector local_coarse, local_fine;
   int dim, n, nd, ne;
public:
   LinearCoarseTransfer(ParFiniteElementSpace &c, ParFiniteElementSpace &f)
      : Operator(f.GetTrueVSize(),c.GetTrueVSize()), coarse(c), fine(f),
        dim(f.GetMesh()->Dimension()), n(f.GetMaxElementOrder()+1), nd(1), ne(f.GetNE())
   { Initialize(); }
   void Initialize()
   {
      auto &c = coarse; auto &f = fine;
      BuildTrueDofMap(f,true_dofs);
      for (int d = 0; d < dim; ++d) { nd *= n; }
      cm = dynamic_cast<const ElementRestriction*>(c.GetElementRestriction(
                                                      ElementDofOrdering::LEXICOGRAPHIC))->GatherMap();
      fm = dynamic_cast<const ElementRestriction*>(f.GetElementRestriction(
                                                      ElementDofOrdering::LEXICOGRAPHIC))->GatherMap();
      H1_SegmentElement segment(n-1); nodes.SetSize(n);
      for (int i = 0; i < n; ++i) { nodes[i] = segment.GetNodes().IntPoint(segment.GetDofMap()[i]).x; }
      Array<int> first(f.GetVSize()); first = std::numeric_limits<int>::max();
      mask.SetSize(ne*nd);
      const int *F = fm.Read(); int *I = first.ReadWrite(), *M = mask.Write();
      mfem::forall(ne*nd, [=] MFEM_HOST_DEVICE (int k) { AtomicMinIndex(I[F[k]],k); });
      mfem::forall(ne*nd, [=] MFEM_HOST_DEVICE (int k) { M[k] = I[F[k]] == k; });
      local_coarse.SetSize(c.GetVSize()); local_fine.SetSize(f.GetVSize());
      local_coarse.UseDevice(true); local_fine.UseDevice(true);
   }
   void Apply(const Vector &x, Vector &y, bool transpose) const
   {
      if (transpose) { RestrictTrueDofs(true_dofs,x,local_fine,true); local_coarse = 0.; }
      else { coarse.GetProlongationMatrix()->Mult(x,local_coarse); local_fine = 0.; }
      const int *C = cm.Read(), *F = fm.Read(), *M = mask.Read();
      const real_t *T = nodes.Read();
      const real_t *X = transpose ? local_fine.Read() : local_coarse.Read();
      real_t *Y = transpose ? local_coarse.ReadWrite() : local_fine.ReadWrite();
      const int dimension = dim, size = n, dofs = nd;
      if (transpose)
      {
         // One block reduces an element's nodal contributions to its vertices.
         // Only 2^dim atomics per cell reach the global coarse vector.
         mfem::forall_2D(ne,128,1,[=] MFEM_HOST_DEVICE (int e)
         {
            MFEM_SHARED real_t sums[8][128];
            MFEM_FOREACH_THREAD(t,x,128)
            {
               real_t sum[8] = {};
               for (int i = t; i < dofs; i += 128)
               {
                  const int k = e*dofs+i;
                  if (!M[k]) { continue; }
                  int a = i; real_t node[3];
                  for (int d = 0; d < dimension; ++d) { node[d] = T[a%size]; a /= size; }
                  const real_t value = X[F[k]];
                  for (int v = 0; v < (1<<dimension); ++v)
                  {
                     real_t weight = value;
                     for (int d = 0; d < dimension; ++d) { weight *= ((v>>d)&1) ? node[d] : 1.-node[d]; }
                     sum[v] += weight;
                  }
               }
               for (int v = 0; v < (1<<dimension); ++v) { sums[v][t] = sum[v]; }
            }
            MFEM_SYNC_THREAD;
            for (int stride = 64; stride > 0; stride /= 2)
            {
               MFEM_FOREACH_THREAD(t,x,stride)
               { for (int v = 0; v < (1<<dimension); ++v) { sums[v][t] += sums[v][t+stride]; } }
               MFEM_SYNC_THREAD;
            }
            MFEM_FOREACH_THREAD(v,x,1<<dimension)
            { AtomicAdd(Y[C[e*(1<<dimension)+v]],sums[v][0]); }
         });
      }
      else
         mfem::forall(ne*nd, [=] MFEM_HOST_DEVICE (int k)
      {
         if (!M[k]) { return; }
         const int e = k/dofs; int i = k%dofs; real_t t[3];
         for (int d = 0; d < dimension; ++d) { t[d] = T[i%size]; i /= size; }
         real_t value = 0.;
         for (int v = 0; v < (1<<dimension); ++v)
         {
            real_t weight = 1.;
            for (int d = 0; d < dimension; ++d) { weight *= ((v>>d)&1) ? t[d] : 1.-t[d]; }
            const int j = C[e*(1<<dimension)+v];
            value += weight*X[j];
         }
         Y[F[k]] = value;
      });
      y.SetSize(transpose ? width : height); y.UseDevice(true);
      if (transpose) { coarse.GetProlongationMatrix()->MultTranspose(local_coarse,y); }
      else { RestrictTrueDofs(true_dofs,local_fine,y,false); }
   }
   void Mult(const Vector &x, Vector &y) const override { Apply(x,y,false); }
   void MultTranspose(const Vector &x, Vector &y) const override { Apply(x,y,true); }
};
