// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "tetlor_entity.hpp"
#include <algorithm>
#include <limits>
#include <map>

namespace tetlor
{
namespace entity
{
namespace
{

// Closed entity maps are ordered with free interior DOFs first. 'energy'
// stores just the interior rows of the two symmetric Schur complements.
struct Block
{
   int lo, hi;              // Offsets in the LOR and Bernstein DOF lists.
   int nl, nh, il, ih;      // Closed and interior free DOF counts.
   int energy, sample, map; // Offsets in the packed, column-major matrices.
   bool owner;
   MFEM_HOST_DEVICE int EnergySize() const { return il*nl + ih*nh; }
   MFEM_HOST_DEVICE int MapSize() const { return ih*(nl + nh-ih); }
};

int Reserve(int &size, long long count)
{
   MFEM_VERIFY(count <= std::numeric_limits<int>::max()-size,
               "Entity transfer exceeds the local index range.");
   const int start = size;
   size += count;
   return start;
}

class DeviceTransfer : public Operator
{
public:
   Array<Block> blocks;
   Array<int> low, high;
   int first[5] = {};
   int forward_lanes[4] = {}, transpose_lanes[4] = {};
   Vector maps;
   mutable Vector adjoint;

   DeviceTransfer(int nh, int nl) : Operator(nh,nl), adjoint(nh)
   { adjoint.UseDevice(true); }

   MemoryClass GetMemoryClass() const override { return Device::GetMemoryClass(); }

   void Apply(const Vector &x, Vector &y, bool transpose) const
   {
      y.UseDevice(true);
      y = 0.0;
      if (transpose) { adjoint = x; }
      const auto b = blocks.Read();
      const auto ld = low.Read(), hd = high.Read();
      const auto m = maps.Read(), input = x.Read();
      auto out = y.ReadWrite();
      auto trace = transpose ? adjoint.ReadWrite() : out;
      for (int stage = 0; stage < 4; ++stage)
      {
         const int d = transpose ? 3-stage : stage;
         const int begin = first[d], n = first[d+1]-begin;
         if (!n) { continue; }
         if (!d)
         {
            mfem::forall(n,[=] MFEM_HOST_DEVICE (int i)
            {
               const Block &v = b[i];
               if (transpose) { out[ld[v.lo]] += trace[hd[v.hi]]; }
               else { trace[hd[v.hi]] = input[ld[v.lo]]; }
            });
            continue;
         }
         // Give small edge/face maps only a few lanes. Several independent
         // entities then share one warp even at low polynomial degree.
         const int lanes = transpose ? transpose_lanes[d] : forward_lanes[d];
         mfem::forall(n*lanes,[=] MFEM_HOST_DEVICE (int thread)
         {
            const int lane = thread%lanes;
            const Block t = b[begin+thread/lanes];
            const int cols = t.nl+t.nh-t.ih;
            const auto a = m+t.map;
            if (transpose)
            {
               for (int j = lane; j < cols; j += lanes)
               {
                  real_t sum = 0;
                  for (int k = 0; k < t.ih; ++k)
                  { sum += a[k+t.ih*j]*trace[hd[t.hi+k]]; }
                  if (j < t.nl) { AtomicAdd(out[ld[t.lo+j]],sum); }
                  else { AtomicAdd(trace[hd[t.hi+t.ih+j-t.nl]],sum); }
               }
            }
            else
            {
               for (int k = lane; k < t.ih; k += lanes)
               {
                  real_t sum = 0;
                  for (int j = 0; j < t.nl; ++j)
                  { sum += a[k+t.ih*j]*input[ld[t.lo+j]]; }
                  for (int j = t.nl; j < cols; ++j)
                  { sum += a[k+t.ih*j]*trace[hd[t.hi+t.ih+j-t.nl]]; }
                  trace[hd[t.hi+k]] = sum;
               }
            }
         });
      }
   }

   void Mult(const Vector &x, Vector &y) const override { Apply(x,y,false); }
   void MultTranspose(const Vector &x, Vector &y) const override { Apply(x,y,true); }
};

// One incident cell, with the target entity first in each free-DOF permutation.
struct Job
{
   int e, block, lo, hi, nl, nh;
   bool sample;
};

// One thread block per incident (cell,entity). Keep the free entity first,
// impose essential values by omission and eliminate the free complement.
// Small cell matrices fit in shared memory. Larger ones use bounded scratch.
template <int Shared>
void AddForms(const Array<Job> &jobs, const Array<int> &perms,
              const Array<Block> &blocks, const DenseTensor &basis,
              const DenseMatrix &metrics, bool high, Vector &forms, int *failed)
{
   const int nd = basis.SizeI();
   Vector scratch(Shared ? 0 : jobs.Size()*nd*nd,Device::GetDeviceMemoryType());
   auto work = scratch.Write();
   const auto job = jobs.Read();
   const auto b = blocks.Read();
   const auto perm = perms.Read();
   const auto g = metrics.Read(), ref = basis.Read();
   auto result = forms.ReadWrite();
   mfem::forall_2D(jobs.Size(),128,1,[=] MFEM_HOST_DEVICE (int id)
   {
      MFEM_SHARED real_t local[Shared ? Shared*Shared : 1];
      real_t *K = Shared ? local : work+id*nd*nd;
      const Job j = job[id];
      const Block t = b[j.block];
      const int n = high ? j.nh : j.nl, keep = high ? t.nh : t.nl;
      const int rows = high ? t.ih : t.il;
      const auto p = perm+(high ? j.hi : j.lo);
      real_t *sum = result+t.energy+(high ? t.il*t.nl : 0);
      const int r = keep == n ? rows : n;
      MFEM_FOREACH_THREAD(k,x,r*n)
      {
         const int i = k%r, col = k/r;
         const int s = p[i]+nd*p[col];
         real_t val = 0.0;
         for (int a = 0; a < 6; ++a) { val += g[a+6*j.e]*ref[s+nd*nd*a]; }
         if (keep == n) { AtomicAdd(sum[k],val); }
         else { K[i+n*col] = val; }
      }
      if (keep < n)
      {
         MFEM_SYNC_THREAD;
         for (int k = n-1; k >= keep; --k)
         {
            const real_t pivot = K[k+n*k];
            if (!(pivot > 0) && MFEM_THREAD_ID(x) == 0) { AtomicAdd(*failed,1); }
            MFEM_FOREACH_THREAD(idx,x,k*k)
            {
               const int i = idx%k, col = idx/k;
               K[i+n*col] -= K[i+n*k]*(K[col+n*k]/pivot);
            }
            MFEM_SYNC_THREAD;
         }
         MFEM_FOREACH_THREAD(k,x,rows*keep)
         { AtomicAdd(sum[k],K[k%rows+n*(k/rows)]); }
      }
   });
}

// Larger cell matrices exceed shared memory. Factor the free complement
// with batched BLAS, solve only for the interior columns, then form the
// rectangular Schur update. Identity padding handles different BC masks.
void AddLargeForms(const Array<Job> &jobs, const Array<int> &perms,
                   const Array<Block> &blocks, const DenseTensor &basis,
                   const DenseMatrix &metrics, bool high, Vector &forms)
{
   Array<int> trace;
   int eliminated = 0, kept = 0, rows = 0;
   for (int i = 0; i < jobs.Size(); ++i)
   {
      const Job &j = jobs[i];
      const Block &b = blocks[j.block];
      const int n = high ? j.nh : j.nl, k = high ? b.nh : b.nl;
      if (n == k) { continue; }
      trace.Append(i);
      eliminated = std::max(eliminated,n-k);
      kept = std::max(kept,k);
      rows = std::max(rows,high ? b.ih : b.il);
   }
   const int nd = basis.SizeI(), count = trace.Size();
   const auto job = jobs.Read();
   const auto blocks_d = blocks.Read();
   const auto p = perms.Read();
   const auto ref = basis.Read(), g = metrics.Read();
   auto dst = forms.ReadWrite();
   auto stiffness = [=] MFEM_HOST_DEVICE (int i, int j, int e)
   {
      real_t x = 0;
      for (int a = 0; a < 6; ++a) { x += g[a+6*e]*ref[i+nd*(j+nd*a)]; }
      return x;
   };
   mfem::forall_2D(jobs.Size(),128,1,[=] MFEM_HOST_DEVICE (int id)
   {
      const Job j = job[id];
      const Block b = blocks_d[j.block];
      const int k = high ? b.nh : b.nl, r = high ? b.ih : b.il;
      const auto perm = p+(high ? j.hi : j.lo);
      auto sum = dst+b.energy+(high ? b.il*b.nl : 0);
      MFEM_FOREACH_THREAD(i,x,k*r)
      { AtomicAdd(sum[i],stiffness(perm[i%r],perm[i/r],j.e)); }
   });
   if (!count) { return; }
   const auto mt = Device::GetDeviceMemoryType();
   DenseTensor A(eliminated,eliminated,count,mt), B(eliminated,kept,count,mt);
   Vector rhs(eliminated*rows*count,mt), update(kept*rows*count,mt);
   const auto ids = trace.Read();
   auto a = A.Write(), b = B.Write(), x = rhs.Write();
   mfem::forall_2D(count,128,1,[=] MFEM_HOST_DEVICE (int id)
   {
      const Job j = job[ids[id]];
      const Block t = blocks_d[j.block];
      const int k = high ? t.nh : t.nl, r = high ? t.ih : t.il;
      const int n = (high ? j.nh : j.nl)-k;
      const auto perm = p+(high ? j.hi : j.lo);
      MFEM_FOREACH_THREAD(idx,x,eliminated*eliminated)
      {
         const int i = idx%eliminated, col = idx/eliminated;
         a[id*eliminated*eliminated+idx] = (i < n && col < n) ?
                                           stiffness(perm[k+i],perm[k+col],j.e) : real_t(i == col);
      }
      MFEM_FOREACH_THREAD(idx,x,eliminated*kept)
      {
         const int i = idx%eliminated, col = idx/eliminated;
         const real_t v = (i < n && col < k) ? stiffness(perm[k+i],perm[col],j.e) : 0;
         b[id*eliminated*kept+idx] = v;
         if (col < rows) { x[id*eliminated*rows+idx] = col < r ? v : 0; }
      }
   });
   const auto &blas = BatchedLinAlg::Get(BatchedLinAlg::GPU_BLAS);
   Array<int> pivot;
   blas.LUFactor(A,pivot);
   blas.LUSolve(A,pivot,rhs);
   blas.MultTranspose(B,rhs,update);
   const auto correction = update.Read();
   mfem::forall_2D(count,64,1,[=] MFEM_HOST_DEVICE (int id)
   {
      const Block t = blocks_d[job[ids[id]].block];
      const int k = high ? t.nh : t.nl, r = high ? t.ih : t.il;
      auto sum = dst+t.energy+(high ? t.il*t.nl : 0);
      MFEM_FOREACH_THREAD(idx,x,k*r)
      { AtomicAdd(sum[idx],-correction[idx/r+kept*(idx%r+rows*id)]); }
   });
}

struct CellJobs
{
   Array<Job> jobs;
   Array<int> perms;
   void Flush(const Array<Block> &blocks, const DenseTensor &lo,
              const DenseTensor &hi, const DenseMatrix &metrics,
              const DenseMatrix &sampling, Vector &forms, Vector &samples,
              int *failed)
   {
      if (!jobs.Size()) { return; }
      for (bool h : {false,true})
      {
         const DenseTensor &ref = h ? hi : lo;
         if (ref.SizeI() <= 32)
         { AddForms<32>(jobs,perms,blocks,ref,metrics,h,forms,failed); }
         else if (ref.SizeI() <= 64)
         { AddForms<64>(jobs,perms,blocks,ref,metrics,h,forms,failed); }
         else if (BatchedLinAlg::IsAvailable(BatchedLinAlg::GPU_BLAS))
         { AddLargeForms(jobs,perms,blocks,ref,metrics,h,forms); }
         else { AddForms<0>(jobs,perms,blocks,ref,metrics,h,forms,failed); }
      }
      const auto j = jobs.Read();
      const auto b = blocks.Read();
      const auto p = perms.Read();
      const auto ref = sampling.Read();
      const int nl = sampling.Height();
      auto J = samples.ReadWrite();
      mfem::forall_2D(jobs.Size(),32,1,[=] MFEM_HOST_DEVICE (int e)
      {
         const Job job = j[e];
         if (job.sample)
         {
            const Block t = b[job.block];
            MFEM_FOREACH_THREAD(k,x,t.il*t.ih)
            { J[t.sample+k] = ref[p[job.lo+k%t.il]+nl*p[job.hi+k/t.il]]; }
         }
      });
      // The next batch replaces all entries. Invalidate the device copies.
      jobs.HostWrite(); perms.HostWrite();
      jobs.SetSize(0); perms.SetSize(0);
   }
};

// Cooperative Cholesky of the interior block, followed by independent
// triangular solves for the columns of the harmonic lift.
MFEM_HOST_DEVICE void FactorLift(real_t *a, int n, int cols, int *failed)
{
   for (int j = 0; j < n; ++j)
   {
      const real_t pivot = a[j+n*j];
      if (!(pivot > 0) && MFEM_THREAD_ID(x) == 0) { AtomicAdd(*failed,1); }
      const real_t diag = sqrt(pivot);
      // All lanes must see the pivot before it is overwritten.
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(k,x,n-j)
      {
         const int i = j+k;
         a[i+n*j] = i == j ? diag : a[i+n*j]/diag;
      }
      MFEM_SYNC_THREAD;
      const int rem = n-j-1;
      MFEM_FOREACH_THREAD(k,x,rem*rem)
      {
         const int i = j+1+k%rem, col = j+1+k/rem;
         if (i >= col) { a[i+n*col] -= a[i+n*j]*a[col+n*j]; }
      }
      MFEM_SYNC_THREAD;
   }
   MFEM_FOREACH_THREAD(col,x,cols-n)
   {
      auto x = a+n*(col+n);
      for (int i = 0; i < n; ++i)
      {
         real_t v = -x[i];
         for (int k = 0; k < i; ++k) { v -= a[i+n*k]*x[k]; }
         x[i] = v/a[i+n*i];
      }
      for (int i = n-1; i >= 0; --i)
      {
         real_t v = x[i];
         for (int k = i+1; k < n; ++k) { v -= a[k+n*i]*x[k]; }
         x[i] = v/a[i+n*i];
      }
   }
   MFEM_SYNC_THREAD;
}

MFEM_HOST_DEVICE void RotateColumns(real_t *a, int n, int i, int j,
                                    real_t c, real_t s)
{
   for (int k = 0; k < n; ++k)
   {
      const real_t x = a[k+n*i], y = a[k+n*j];
      a[k+n*i] = c*x-s*y;
      a[k+n*j] = s*x+c*y;
   }
}

// One-sided Jacobi SVD on the small, rectangular whitened sampling matrix.
// Return its transposed polar factor, retaining the relative accuracy of the
// small singular values (forming the normal equations would square kappa).
MFEM_HOST_DEVICE void PolarTranspose(real_t *a, int m, int n, real_t *r,
                                     int *failed)
{
   real_t *v = a+m*n;
   MFEM_FOREACH_THREAD(k,x,n*n) { v[k] = k%n == k/n ? 1 : 0; }
   MFEM_SYNC_THREAD;
   const real_t tol = 8*std::numeric_limits<real_t>::epsilon();
   // Round-robin Jacobi ordering pairs every two columns once per sweep.
   // Each round consists of disjoint plane rotations.
   const int last = n+(n%2)-1;
   MFEM_SHARED int changed;
   for (int sweep = 0; n > 1; ++sweep)
   {
      if (MFEM_THREAD_ID(x) == 0) { changed = 0; }
      MFEM_SYNC_THREAD;
      for (int round = 0; round < last; ++round)
      {
         MFEM_FOREACH_THREAD(pair,x,(last+1)/2)
         {
            const int i = (round+pair)%last;
            const int j = pair ? (round+last-pair)%last : last;
            if (j == n) { continue; }
            real_t aa = 0, bb = 0, ab = 0;
            for (int k = 0; k < m; ++k)
            {
               const real_t x = a[k+m*i], y = a[k+m*j];
               aa += x*x; bb += y*y; ab += x*y;
            }
            if (fabs(ab) <= tol*sqrt(aa)*sqrt(bb)) { continue; }
            AtomicAdd(changed,1);
            const real_t tau = (bb-aa)/(2*ab);
            const real_t t = copysign(real_t(1),tau)/(fabs(tau)+hypot(real_t(1),tau));
            const real_t c = 1/hypot(real_t(1),t);
            RotateColumns(a,m,i,j,c,c*t);
            RotateColumns(v,n,i,j,c,c*t);
         }
         MFEM_SYNC_THREAD;
      }
      const bool done = changed == 0;
      MFEM_SYNC_THREAD;
      if (done) { break; }
      if (sweep == 100)
      {
         if (MFEM_THREAD_ID(x) == 0) { AtomicAdd(*failed,1); }
         break;
      }
   }
   MFEM_FOREACH_THREAD(j,x,n)
   {
      real_t norm = 0;
      for (int i = 0; i < m; ++i) { norm += a[i+m*j]*a[i+m*j]; }
      if (!(norm > 0)) { AtomicAdd(*failed,1); }
      const real_t inv = 1/sqrt(norm);
      for (int i = 0; i < m; ++i) { a[i+m*j] *= inv; }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(idx,x,m*n)
   {
      const int i = idx%n, j = idx/n;
      real_t sum = 0;
      for (int k = 0; k < n; ++k) { sum += v[i+n*k]*a[j+m*k]; }
      r[idx] = sum;
   }
   MFEM_SYNC_THREAD;
}

void BuildMaps(DeviceTransfer &op, Vector &forms, Vector &samples, int *failed)
{
   const auto b = op.blocks.Read();
   auto energies = forms.ReadWrite(), sampling = samples.ReadWrite();
   auto maps = op.maps.Write();
   const int first = op.first[1];
   mfem::forall_2D(op.blocks.Size()-first,32,1,[=] MFEM_HOST_DEVICE (int id)
   {
      const Block t = b[first+id];
      if (!t.owner) { return; }
      const int m = t.il, n = t.ih;
      auto l = energies+t.energy, h = l+m*t.nl;
      auto a = sampling+t.sample, r = maps+t.map;
      FactorLift(l,m,t.nl,failed);
      FactorLift(h,n,t.nh,failed);
      MFEM_FOREACH_THREAD(j,x,n)
      {
         for (int i = 0; i < m; ++i)
         {
            real_t x = 0;
            for (int k = i; k < m; ++k) { x += l[k+m*i]*a[k+m*j]; }
            a[i+m*j] = x;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i,x,m)
      {
         for (int j = 0; j < n; ++j)
         {
            real_t x = a[i+m*j];
            for (int k = 0; k < j; ++k) { x -= h[j+n*k]*a[i+m*k]; }
            a[i+m*j] = x/h[j+n*j];
         }
      }
      MFEM_SYNC_THREAD;
      PolarTranspose(a,m,n,r,failed);
      MFEM_FOREACH_THREAD(i,x,n)
      {
         for (int j = m-1; j >= 0; --j)
         {
            real_t x = 0;
            for (int k = 0; k <= j; ++k) { x += r[i+n*k]*l[j+m*k]; }
            r[i+n*j] = x;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(j,x,m)
      {
         for (int i = n-1; i >= 0; --i)
         {
            real_t x = r[i+n*j];
            for (int k = i+1; k < n; ++k) { x -= h[k+n*i]*r[k+n*j]; }
            r[i+n*j] = x/h[i+n*i];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(idx,x,n*(t.nl-m))
      {
         const int i = idx%n, j = m+idx/n;
         real_t x = 0;
         for (int k = 0; k < m; ++k) { x -= r[i+n*k]*l[k+m*j]; }
         r[i+n*j] = x;
      }
      MFEM_FOREACH_THREAD(idx,x,n*(t.nh-n)) { r[n*t.nl+idx] = h[n*n+idx]; }
   });
}

#ifdef MFEM_USE_MPI
// Pack only shared entries. The true-DOF table lets GroupCommunicator use
// its device gather/reduction even though these DOFs represent matrix entries.
void Exchange(ParMesh &mesh, const Array<int> ids[], const Array<Block> &blocks,
              Vector &values, bool reduce)
{
   Array<int> indices, groups, truth;
   int owned = 0;
   for (int d : {1,2})
   {
      for (int g = 1; g < mesh.GetNGroups(); ++g)
      {
         const bool master = mesh.gtopo.IAmMaster(g);
         const int count = d == 1 ? mesh.GroupNEdges(g) : mesh.GroupNTriangles(g);
         for (int j = 0; j < count; ++j)
         {
            const int id = d == 1 ? mesh.GroupEdge(g,j) : mesh.GroupTriangle(g,j);
            const int i = ids[d][id];
            if (i < 0) { continue; }
            const Block &b = blocks[i];
            const int n = reduce ? b.EnergySize() : b.MapSize();
            const int start = reduce ? b.energy : b.map;
            for (int k = 0; k < n; ++k)
            {
               indices.Append(start+k);
               groups.Append(g);
               truth.Append(master ? owned++ : -1);
            }
         }
      }
   }
   GroupCommunicator comm(mesh.gtopo);
   comm.Create(groups);
   comm.SetLTDofTable(truth);
   Array<real_t> buffer(indices.Size());
   buffer.UseDevice(true);
   const auto idx = indices.Read(), owner = truth.Read();
   auto v = values.ReadWrite(), buf = buffer.Write();
   mfem::forall(indices.Size(),[=] MFEM_HOST_DEVICE (int i)
   { buf[i] = reduce || owner[i] >= 0 ? v[idx[i]] : 0; });
   if (reduce) { comm.Reduce(buffer,GroupCommunicator::Sum); }
   else { comm.Bcast(buffer); }
   const auto result = buffer.Read();
   mfem::forall(indices.Size(),[=] MFEM_HOST_DEVICE (int i)
   { v[idx[i]] = result[i]; });
}
#endif

void ReferenceTables(BatchedLOR &lor, const FiniteElement &bern,
                     DenseTensor &low, DenseTensor &high, DenseMatrix &sample)
{
   IsoparametricTransformation T;
   T.SetIdentityTransformation(Geometry::TETRAHEDRON);
   int k = 0;
   for (int i = 0; i < 3; ++i)
   {
      for (int j = i; j < 3; ++j,++k)
      {
         real_t metric[6] = {};
         metric[k] = 1;
         lor.GetPatchStiffness(metric,low(k));
         DenseMatrix G(3);
         G = 0.0;
         G(i,j) = G(j,i) = 1;
         MatrixConstantCoefficient coefficient(G);
         DiffusionIntegrator(coefficient).AssembleElementMatrix(bern,T,high(k));
      }
   }
   Vector shape(bern.GetDof());
   for (int i = 0; i < sample.Height(); ++i)
   {
      IntegrationPoint ip;
      ip.Set3(lor.ReferencePatch().GetVertex(i));
      bern.CalcShape(ip,shape);
      sample.SetRow(i,shape);
   }
}

void AddPermutation(Array<int> &perm, const int *entity, int keep,
                    const Array<int> &index, const Array<int> &free,
                    Array<int> &mask)
{
   mask = 0;
   for (int i = 0; i < keep; ++i)
   {
      const int k = index[entity[i]];
      mask[k] = 1;
      perm.Append(k);
   }
   for (int k : free) { if (!mask[k]) { perm.Append(k); } }
}

void AssembleForms(const DeviceTransfer &op, FiniteElementSpace &high,
                   BatchedLOR &lor, const Array<int> ids[],
                   const Array<int> &eh, const Array<int> &el,
                   Vector &forms, Vector &samples, int *failed)
{
   const FiniteElement &bern = *high.GetFE(0);
   const int nl = lor.ReferencePatch().GetNV(), nh = bern.GetDof();
   DenseTensor L(nl,nl,6), H(nh,nh,6);
   DenseMatrix J(nl,nh);
   ReferenceTables(lor,bern,L,H,J);
   const auto labels = BernsteinLabels(bern);
   std::map<BarycentricLabel,int> number;
   for (int i = 0; i < nh; ++i) { number[labels[i]] = i; }
   Mesh &mesh = *high.GetMesh();
   Array<int> hi_index(op.Height()), lo_index(op.Width()), hl(nh), ll(nl);
   Array<int> free_hi, free_lo, hd, edges, faces, orientation;
   Array<int> sampled(op.blocks.Size());
   sampled = 0;
   const auto patch = lor.PatchDofs().HostRead();
   const auto high_ess = eh.HostRead(), low_ess = el.HostRead();
   // A batch is bounded by both its permutation data and the largest global
   // scratch matrix. Shared-memory kernels need no per-job global scratch.
   const int nd = std::max(nh,nl);
   const int limit = std::max(1, (4*1024*1024)/(nd > 64 ? nd*nd : 2*nd));
   CellJobs batch;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      free_hi.SetSize(0); free_lo.SetSize(0);
      high.GetElementDofs(e,hd);
      const auto perm = lor.VertexPermutation(e);
      for (int i = 0; i < nh; ++i)
      {
         const auto &a = labels[i];
         const int d = hd[number.at({a[perm[0]],a[perm[1]],a[perm[2]],a[perm[3]]})];
         hi_index[d] = i;
         if (!high_ess[d]) { free_hi.Append(i); }
      }
      for (int i = 0; i < nl; ++i)
      {
         const int d = patch[e*nl+i];
         lo_index[d] = i;
         if (!low_ess[d]) { free_lo.Append(i); }
      }
      auto add = [&](int id)
      {
         if (id < 0) { return; }
         const Block &b = op.blocks[id];
         const int lo = batch.perms.Size();
         AddPermutation(batch.perms,op.low.GetData()+b.lo,b.nl,lo_index,free_lo,ll);
         const int hi = batch.perms.Size();
         AddPermutation(batch.perms,op.high.GetData()+b.hi,b.nh,hi_index,free_hi,hl);
         batch.jobs.Append(Job{e,id,lo,hi,free_lo.Size(),free_hi.Size(),!sampled[id]});
         sampled[id] = 1;
         if (batch.jobs.Size() == limit)
         { batch.Flush(op.blocks,L,H,lor.PatchMetrics(),J,forms,samples,failed); }
      };
      mesh.GetElementEdges(e,edges,orientation);
      mesh.GetElementFaces(e,faces,orientation);
      for (int id : edges) { add(ids[1][id]); }
      for (int id : faces) { add(ids[2][id]); }
      add(ids[3][e]);
   }
   batch.Flush(op.blocks,L,H,lor.PatchMetrics(),J,forms,samples,failed);
}

} // namespace

std::unique_ptr<Operator> MakeDeviceTransfer(FiniteElementSpace &high,
                                             BatchedLOR &lor)
{
   auto op = std::make_unique<DeviceTransfer>(high.GetVSize(),
                                              lor.Space().GetVSize());
   Mesh &mesh = *high.GetMesh();
   const auto eh = BoundaryMarker(high), el = BoundaryMarker(lor.Space());
   const int counts[] = {mesh.GetNV(),mesh.GetNEdges(),mesh.GetNFaces(),mesh.GetNE()};
   const Geometry::Type geom[] = {Geometry::POINT,Geometry::SEGMENT,
                                  Geometry::TRIANGLE,Geometry::TETRAHEDRON
                                 };
   Array<int> ids[4];
   int energies = 0, samples = 0, maps = 0;
   for (int d = 0; d < 4; ++d)
   {
      op->first[d] = op->blocks.Size();
      ids[d].SetSize(counts[d]); ids[d] = -1;
      const int interior = high.FEColl()->DofForGeometry(geom[d]);
      if (!interior) { continue; }
      auto lanes = [](int n)
      {
         int w = 1;
         while (w < n && w < 32) { w *= 2; }
         return w;
      };
      op->forward_lanes[d] = lanes(interior);
      int cols = 0;
      for (int i = 0; i < counts[d]; ++i)
      {
         EntityDofs h,l;
         h.Init(high,d,i,eh);
         if (!h.interior) { continue; }
         l.Init(lor.Space(),d,i,el);
         Block b{};
         b.lo = op->low.Size(); b.hi = op->high.Size();
         b.il = l.interior; b.ih = h.interior;
         b.nl = l.all.Size(); b.nh = h.all.Size();
         b.owner = true;
         cols = std::max(cols,b.nl+b.nh-b.ih);
         if (d)
         {
            b.energy = Reserve(energies,1LL*b.il*b.nl + 1LL*b.ih*b.nh);
            b.sample = Reserve(samples,1LL*b.ih*(b.il+b.ih));
            b.map = Reserve(maps,1LL*b.ih*(b.nl+b.nh-b.ih));
         }
         ids[d][i] = op->blocks.Size();
         op->blocks.Append(b);
         op->low.Append(l.all); op->high.Append(h.all);
      }
      op->transpose_lanes[d] = lanes(cols);
   }
   op->first[4] = op->blocks.Size();
#ifdef MFEM_USE_MPI
   auto pm = dynamic_cast<ParMesh*>(&mesh);
   if (pm)
   {
      for (int d : {1,2})
      {
         for (int g = 1; g < pm->GetNGroups(); ++g)
         {
            const int count = d == 1 ? pm->GroupNEdges(g) : pm->GroupNTriangles(g);
            for (int j = 0; j < count; ++j)
            {
               const int id = d == 1 ? pm->GroupEdge(g,j) : pm->GroupTriangle(g,j);
               if (ids[d][id] >= 0) { op->blocks[ids[d][id]].owner = pm->gtopo.IAmMaster(g); }
            }
         }
      }
   }
#endif
   const auto mt = Device::GetDeviceMemoryType();
   op->maps.SetSize(maps,mt);
   if (!maps) { return op; }
   Vector forms(energies,mt), sampling(samples,mt);
   forms.UseDevice(true); forms = 0.0;
   Array<int> error(1);
   error = 0;
   int *failed = error.ReadWrite();
   AssembleForms(*op,high,lor,ids,eh,el,forms,sampling,failed);
#ifdef MFEM_USE_MPI
   if (pm) { Exchange(*pm,ids,op->blocks,forms,true); }
#endif
   BuildMaps(*op,forms,sampling,failed);
   MFEM_VERIFY(error.HostRead()[0] == 0,
               "Singular entity energy or sampling matrix.");
#ifdef MFEM_USE_MPI
   if (pm) { Exchange(*pm,ids,op->blocks,op->maps,false); }
#endif
   return op;
}

} // namespace entity
} // namespace tetlor
