// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE and NOTICE for details.

// Included inside the vertex_patch implementation namespace.
namespace
{
// Rectangular contractions, batched over cells, with runtime-sized workspace.
void TensorBatch(const Vector &x, Vector &y,
                 const std::vector<const DenseMatrix*> &matrices, int ne)
{
   Vector u(x), v; u.UseDevice(true); v.UseDevice(true);
   std::vector<int> sizes;
   for (const auto *m : matrices) { sizes.push_back(m->Width()); }
   for (int axis = 0; axis < int(matrices.size()); ++axis)
   {
      const auto &m = *matrices[axis];
      int before = 1, after = 1;
      for (int d = 0; d < axis; ++d) { before *= sizes[d]; }
      for (int d = axis+1; d < int(sizes.size()); ++d) { after *= sizes[d]; }
      const int ni = sizes[axis], no = m.Height();
      const int src_size = before*ni*after, dst_size = before*no*after;
      v.SetSize(ne*dst_size);
      const real_t *U = u.Read(), *M = m.Read(); real_t *V = v.Write();
      mfem::forall(v.Size(), [=] MFEM_HOST_DEVICE (int k)
      {
         const int e = k/dst_size, i = k%before;
         const int j = (k/before)%no, outer = (k%dst_size)/(before*no);
         real_t value = 0.;
         for (int l = 0; l < ni; ++l)
         { value += M[j+no*l]*U[e*src_size+i+before*(l+ni*outer)]; }
         V[k] = value;
      });
      sizes[axis] = no; u.Swap(v);
   }
   y = u;
}

// One reference sparsity template per degree/dimension; numerical values for
// all cells live in contiguous arrays. No dense high-order cell matrix.
class ElementBatch
{
   Array<int> term_offsets, term_diagonal, uncondensed_source;
   Vector term_weight;
   Array<int> schur_offsets, schur_a, schur_b, schur_d, schur_copy;
   DenseMatrix values, derivatives;
public:
   IntegrationRules rules{0, Quadrature1D::GaussLobatto};
   std::unique_ptr<SparseMatrix> full_pattern, pattern;
   std::vector<int> lex;
   Array<int> lex_map, entry_rows, cross_rows, cross_cols, cross_sources,
         interior_dofs, interior_sources;
   Vector full_values, active_values;
   int dim, p, n, nz, active_nz;
   ElementBatch(const Basis &basis, int dimension, bool condensed)
      : dim(dimension), p(basis.p), n(p+1)
   {
      basis.S.HostRead(); basis.G.HostRead(); basis.derivative.HostRead();
      basis.eigenvalues.HostRead();
      const auto &ir = rules.Get(Geometry::SEGMENT, 3*p);
      const int nq = ir.GetNPoints();
      H1_SegmentElement fe(p); const auto &map = fe.GetDofMap();
      DenseMatrix gi(n), coeff(n), dshape(n,1); Vector shape(n);
      DenseMatrixInverse(basis.G).GetInverseMatrix(gi);
      Mult(basis.S, gi, coeff);
      values.SetSize(n,nq); derivatives.SetSize(p,nq);
      for (int q = 0; q < nq; ++q)
      {
         fe.CalcShape(ir.IntPoint(q), shape); fe.CalcDShape(ir.IntPoint(q), dshape);
         for (int i = 0; i < n; ++i)
         {
            real_t v = 0.;
            for (int j = 0; j < n; ++j) { v += shape[map[j]]*coeff(j,i); }
            values(i,q) = v*v;
         }
         derivatives(0,q) = 1.;
         for (int i = 1; i < p; ++i)
         {
            real_t v = 0.;
            for (int j = 0; j < n; ++j) { v += dshape(map[j],0)*basis.S(j,i); }
            derivatives(i,q) = v*v/basis.eigenvalues[i-1];
         }
      }
      SparseMatrix mass(n), derivative(p,n);
      for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
         {
            if (basis.G(i,j) != 0.) { mass.Add(i,j,basis.G(i,j)); }
            if (i < p && basis.derivative(i,j) != 0.)
            { derivative.Add(i,j,basis.derivative(i,j)); }
         }
      mass.Finalize(); derivative.Finalize();
      using Entry = std::pair<int,int>;
      std::map<Entry,std::vector<std::pair<int,real_t>>> terms;
      const int diagonal_size = p*Power(n,dim-1), nd = Power(n,dim);
      for (int axis = 0; axis < dim; ++axis)
      {
         auto b = std::make_unique<SparseMatrix>(axis == 0 ? derivative : mass);
         for (int d = 1; d < dim; ++d)
         { b.reset(OuterProduct(d == axis ? derivative : mass, *b)); }
         const int *bi = b->HostReadI(), *bj = b->HostReadJ();
         const real_t *bv = b->HostReadData();
         for (int r = 0; r < b->Height(); ++r)
            for (int i = bi[r]; i < bi[r+1]; ++i)
               for (int j = bi[r]; j < bi[r+1]; ++j)
               { terms[ {bj[i],bj[j]}].emplace_back(axis*diagonal_size+r,bv[i]*bv[j]); }
      }
      full_pattern = std::make_unique<SparseMatrix>(nd);
      for (const auto &entry : terms)
      {
         real_t v = 0.;
         for (const auto &t : entry.second) { v += t.second; }
         full_pattern->Add(entry.first.first,entry.first.second,v);
      }
      full_pattern->Finalize(0); full_pattern->SortColumnIndices();
      nz = full_pattern->NumNonZeroElems();
      std::map<Entry,int> position;
      std::vector<int> offsets{0}, indices; std::vector<real_t> weights;
      const int *ai = full_pattern->HostReadI(), *aj = full_pattern->HostReadJ();
      for (int i = 0; i < nd; ++i) for (int k = ai[i]; k < ai[i+1]; ++k)
         {
            position[ {i,aj[k]}] = k;
            for (const auto &t : terms.at({i,aj[k]}))
            { indices.push_back(t.first); weights.push_back(t.second); }
            offsets.push_back(indices.size());
         }
      CopyArray(offsets,term_offsets); CopyArray(indices,term_diagonal);
      term_weight.SetSize(weights.size());
      std::copy(weights.begin(),weights.end(),term_weight.HostWrite());
      if (!condensed)
      {
         // Include the Schur fill even when solving the full patch. This is
         // the prescribed ICC graph, generated once on the reference cell.
         std::vector<int> boundary;
         auto schur = CondenseElement(*full_pattern,p,dim,boundary);
         pattern = std::make_unique<SparseMatrix>(nd);
         for (const auto &entry : position) { pattern->Add(entry.first.first,entry.first.second,0.); }
         const int *si = schur->HostReadI(), *sj = schur->HostReadJ();
         for (int i = 0; i < schur->Height(); ++i)
            for (int k = si[i]; k < si[i+1]; ++k) { pattern->Add(boundary[i],boundary[sj[k]],0.); }
         pattern->Finalize(0); pattern->SortColumnIndices();
         const int *pi = pattern->HostReadI(), *pj = pattern->HostReadJ();
         std::vector<int> mapping;
         for (int i = 0; i < nd; ++i) for (int k = pi[i]; k < pi[i+1]; ++k)
            {
               const auto it = position.find({i,pj[k]});
               mapping.push_back(it == position.end() ? -1 : it->second);
            }
         CopyArray(mapping,uncondensed_source);
         lex.resize(nd);
         for (int i = 0; i < nd; ++i) { lex[i] = i; }
      }
      else
      {
         pattern = CondenseElement(*full_pattern,p,dim,lex);
         std::vector<int> boundary_index(nd,-1);
         for (int i = 0; i < int(lex.size()); ++i) { boundary_index[lex[i]] = i; }
         std::map<Entry,std::vector<std::array<int,3>>> updates;
         for (int i = 0; i < nd; ++i) if (boundary_index[i] < 0)
            {
               const int diag = position.at({i,i});
               for (int k = ai[i]; k < ai[i+1]; ++k) if (boundary_index[aj[k]] >= 0)
                     for (int l = ai[i]; l < ai[i+1]; ++l) if (boundary_index[aj[l]] >= 0)
                        { updates[ {boundary_index[aj[k]],boundary_index[aj[l]]}].push_back({k,l,diag}); }
            }
         std::vector<int> so{0}, sa, sb, sd, sc;
         const int *si = pattern->HostReadI(), *sj = pattern->HostReadJ();
         for (int i = 0; i < pattern->Height(); ++i)
            for (int k = si[i]; k < si[i+1]; ++k)
            {
               auto it = position.find({lex[i],lex[sj[k]]});
               sc.push_back(it == position.end() ? -1 : it->second);
               for (const auto &u : updates[ {i,sj[k]}])
               { sa.push_back(u[0]); sb.push_back(u[1]); sd.push_back(u[2]); }
               so.push_back(sa.size());
            }
         CopyArray(so,schur_offsets); CopyArray(sa,schur_a); CopyArray(sb,schur_b);
         CopyArray(sd,schur_d); CopyArray(sc,schur_copy);
      }
      active_nz = pattern->NumNonZeroElems();
      CopyArray(lex,lex_map);
      std::vector<int> er, cr, cc, cs, id, is;
      const int *pi = pattern->HostReadI();
      for (int i = 0; i < pattern->Height(); ++i)
         for (int k = pi[i]; k < pi[i+1]; ++k) { er.push_back(i); }
      for (int i = 0; i < nd; ++i) if (Interior(i,n,dim))
            for (int k = ai[i]; k < ai[i+1]; ++k)
            {
               if (aj[k] == i) { id.push_back(i); is.push_back(k); }
               else if (!Interior(aj[k],n,dim))
               { cr.push_back(aj[k]); cc.push_back(i); cs.push_back(k); }
            }
      CopyArray(er,entry_rows); CopyArray(cr,cross_rows); CopyArray(cc,cross_cols);
      CopyArray(cs,cross_sources); CopyArray(id,interior_dofs);
      CopyArray(is,interior_sources);

   }
   static std::shared_ptr<ElementBatch> Get(const Basis &basis, int dim,
                                            bool condensed)
   {
      static std::map<std::tuple<int,int,bool>,std::shared_ptr<ElementBatch>> cache;
      auto &entry = cache[ {basis.p,dim,condensed}];
      if (!entry) { entry = std::make_shared<ElementBatch>(basis,dim,condensed); }
      return entry;
   }
   void Assemble(ParMesh &mesh, Coefficient &coefficient, bool condensed)
   {
      const int ne = mesh.GetNE(), dimension = dim;
      const auto &ir = rules.Get(TensorBasisElement::GetTensorProductGeometry(dim),
                                 3*p);
      const int nq = ir.GetNPoints(), ds = p*Power(n,dim-1);
      QuadratureSpace qs(mesh, ir); QuadratureFunction alpha(qs);
      coefficient.Project(alpha);
      const auto *geom = mesh.GetGeometricFactors(ir,GeometricFactors::JACOBIANS);
      const auto J = Reshape(geom->J.Read(),nq,dim,dim,ne);
      const real_t *A = alpha.Read();
      Vector weights(nq);
      for (int q = 0; q < nq; ++q) { weights[q] = ir.IntPoint(q).weight; }
      const real_t *W = weights.Read();
      Vector diagonals(ne*dim*ds); diagonals.UseDevice(true);
      Array<int> invalid(1); invalid = 0; int *bad = invalid.ReadWrite();
      for (int axis = 0; axis < dim; ++axis)
      {
         Vector metric(ne*nq), diagonal; metric.UseDevice(true);
         diagonal.UseDevice(true);
         real_t *M = metric.Write();
         mfem::forall(ne*nq, [=] MFEM_HOST_DEVICE (int k)
         {
            const int e = k/nq, q = k%nq;
            real_t jac[9], inv[9];
            for (int c = 0; c < dimension; ++c) for (int r = 0; r < dimension; ++r)
               { jac[r+dimension*c] = J(q,r,c,e); }
            real_t det;
            if (dimension == 2)
            { det = kernels::Det<2>(jac); kernels::CalcInverse<2>(jac,inv); }
            else
            { det = kernels::Det<3>(jac); kernels::CalcInverse<3>(jac,inv); }
            real_t norm = 0.;
            for (int j = 0; j < dimension; ++j) { norm += inv[axis+dimension*j]*inv[axis+dimension*j]; }
            if (!(det > 0.) || !(A[k] > 0.) || !std::isfinite(A[k])) { AtomicAdd(bad[0],1); }
            M[k] = W[q]*det*A[k]*norm;
         });
         std::vector<const DenseMatrix*> matrices(dim,&values);
         matrices[axis] = &derivatives;
         TensorBatch(metric,diagonal,matrices,ne);
         const real_t *D = diagonal.Read(); real_t *all = diagonals.ReadWrite();
         const int axes = dim;
         mfem::forall(ne*ds, [=] MFEM_HOST_DEVICE (int k)
         { all[(k/ds)*axes*ds+axis*ds+k%ds] = D[k]; });
      }
      MFEM_VERIFY(invalid.HostRead()[0] == 0,
                  "Expected positive diffusion and positively oriented elements");
      const int *T = term_offsets.Read(), *Q = term_diagonal.Read();
      const real_t *C = term_weight.Read(), *D = diagonals.Read();
      const int entries = nz, axes = dim;
      full_values.SetSize(ne*nz); full_values.UseDevice(true);
      real_t *F = full_values.Write();
      mfem::forall(ne*nz, [=] MFEM_HOST_DEVICE (int k)
      {
         const int e = k/entries, a = k%entries;
         real_t value = 0.;
         for (int t = T[a]; t < T[a+1]; ++t) { value += C[t]*D[e*axes*ds+Q[t]]; }
         F[k] = value;
      });
      active_values.UseDevice(true);
      if (!condensed)
      {
         active_values.SetSize(ne*active_nz); real_t *A = active_values.Write();
         const int count = active_nz; const int *S = uncondensed_source.Read();
         mfem::forall(ne*count, [=] MFEM_HOST_DEVICE (int k)
         { const int src = S[k%count]; A[k] = src < 0 ? 0. : F[(k/count)*entries+src]; });
         return;
      }
      active_values.SetSize(ne*active_nz); real_t *S = active_values.Write();
      const int snz = active_nz;
      const int *SO = schur_offsets.Read(), *SA = schur_a.Read(),
                 *SB = schur_b.Read();
      const int *SD = schur_d.Read(), *SC = schur_copy.Read();
      mfem::forall(ne*snz, [=] MFEM_HOST_DEVICE (int k)
      {
         const int e = k/snz, a = k%snz, base = e*entries;
         real_t v = SC[a] < 0 ? 0. : F[base+SC[a]];
         for (int t = SO[a]; t < SO[a+1]; ++t)
         { v -= F[base+SA[t]]*F[base+SB[t]]/F[base+SD[t]]; }
         S[k] = v;
      });
   }
};
}
