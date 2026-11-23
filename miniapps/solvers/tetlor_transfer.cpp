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
#include <cmath>
#include <map>


namespace tetlor
{
std::vector<BarycentricLabel> BernsteinLabels(const FiniteElement &bern)
{
   const int p = bern.GetOrder();
   std::vector<BarycentricLabel> alpha(bern.GetDof());
   for (int i = 0; i < bern.GetDof(); ++i)
   {
      const auto &ip = bern.GetNodes().IntPoint(i);
      auto &a = alpha[i];
      a = {0,std::lround(p*ip.x),std::lround(p*ip.y),std::lround(p*ip.z)};
      a[0] = p-a[1]-a[2]-a[3];
   }
   return alpha;
}

namespace
{

// The sampled Bernstein basis J = Q U is orthonormalized with the normalized
// lumped weights. These weights and the polynomial means are affine invariant.
class PatchBasis
{
   DenseMatrix Q, U;
   Vector mean, weights;

public:
   PatchBasis(Mesh &patch, const FiniteElement &bern)
      : Q(patch.GetNV(), bern.GetDof()), U(bern.GetDof()),
        mean(bern.GetDof()), weights(patch.GetNV())
   {
      Vector lump(patch.GetNV());
      lump = 0.0;
      for (int s = 0; s < patch.GetNE(); s++)
      {
         const Element &el = *patch.GetElement(s);
         const real_t weight = patch.GetElementVolume(s)/el.GetNVertices();
         for (int j = 0; j < el.GetNVertices(); j++)
         {
            lump(el.GetVertices()[j]) += weight;
         }
      }
      lump /= lump.Sum();
      Vector root(lump.Size()), shape(bern.GetDof());
      for (int i = 0; i < patch.GetNV(); i++)
      {
         root(i) = std::sqrt(lump(i));
         IntegrationPoint ip;
         ip.Set3(patch.GetVertex(i));
         bern.CalcShape(ip, shape);
         Q.SetRow(i, shape);
      }

      U = 0.0;
      Vector col, prev;
      for (int j = 0; j < Q.Width(); j++)
      {
         Q.GetColumnReference(j, col);
         col *= root;
         for (int pass = 0; pass < 2; pass++)
         {
            for (int i = 0; i < j; i++)
            {
               Q.GetColumnReference(i, prev);
               const real_t coeff = col*prev;
               U(i,j) += coeff;
               col.Add(-coeff, prev);
            }
         }
         U(j,j) = col.Norml2();
         MFEM_VERIFY(U(j,j) > 0.0, "Rank-deficient polynomial sampling.");
         col /= U(j,j);

         // Each Bernstein function has mean 1/nh. Solve U^t mean = 1/nh.
         real_t value = 1.0/Q.Width();
         for (int i = 0; i < j; i++) { value -= U(i,j)*mean(i); }
         mean(j) = value/U(j,j);
      }
      for (int j = 0; j < Q.Width(); j++)
      {
         Q.GetColumnReference(j, col);
         col /= root;
      }
      Q.Mult(mean, weights);
      weights *= lump;
   }

   // Q^t K and its Gram matrix are linear in the six affine metric entries.
   // Tabulate each reference contribution once, then solve the physical patch
   // systems in batches without materializing a stiffness matrix per element.
   void FitDevice(const BatchedLOR &lor, DenseTensor &fits)
   {
      const int nh = Q.Width(), nl = Q.Height(), ne = fits.SizeK();
      DenseTensor rhs_terms(nh,nl,7), gram_terms(nh,nh,7);
      for (int k = 0; k < 6; ++k)
      {
         real_t metric[6] = {};
         metric[k] = 1;
         lor.ApplyPatchStiffness(metric,Q,rhs_terms(k));
         Mult(rhs_terms(k),Q,gram_terms(k));
      }
      rhs_terms(6) = 0.0;
      gram_terms(6) = 0.0;
      AddMult_a_VWt(1.0,mean,weights,rhs_terms(6));
      AddMult_a_VVt(1.0,mean,gram_terms(6));

      const auto metric = Reshape(lor.PatchMetrics().Read(),6,ne);
      Vector scale(ne,Device::GetDeviceMemoryType());
      const auto volume = lor.PatchVolumes().Read();
      auto h = scale.Write();
      mfem::forall(ne,[=] MFEM_HOST_DEVICE (int i) { h[i] = cbrt(volume[i]); });
      const auto u = Reshape(U.Read(),nh,nh);
      const auto rhs_basis = Reshape(rhs_terms.Read(),nh*nl,7);
      const auto gram_basis = Reshape(gram_terms.Read(),nh*nh,7);
      auto result = Reshape(fits.Write(),nh,nl,ne);
      const auto mt = Device::GetDeviceMemoryType();
      const auto backend = BatchedLinAlg::IsAvailable(BatchedLinAlg::GPU_BLAS) ?
                           BatchedLinAlg::GPU_BLAS : BatchedLinAlg::NATIVE;
      const auto &blas = BatchedLinAlg::Get(backend);
      // Bound temporary storage independently of the partition size.
      const int batch = std::max(1, (4*1024*1024)/(nh*(nh+nl)));
      for (int first = 0; first < ne; first += batch)
      {
         const int n = std::min(batch,ne-first);
         DenseTensor grams(nh,nh,n,mt);
         Vector rhs(nh*nl*n,mt);
         auto g = Reshape(grams.Write(),nh*nh,n);
         auto r = Reshape(rhs.Write(),nh*nl,n);
         mfem::forall(nh*nh,n,[=] MFEM_HOST_DEVICE (int i, int e)
         {
            real_t v = h[first+e]*gram_basis(i,6);
            for (int k = 0; k < 6; ++k) { v += metric(k,first+e)*gram_basis(i,k); }
            g(i,e) = v;
         });
         mfem::forall(nh*nl,n,[=] MFEM_HOST_DEVICE (int i, int e)
         {
            real_t v = h[first+e]*rhs_basis(i,6);
            for (int k = 0; k < 6; ++k) { v += metric(k,first+e)*rhs_basis(i,k); }
            r(i,e) = v;
         });
         Array<int> pivots;
         blas.LUFactor(grams,pivots);
         blas.LUSolve(grams,pivots,rhs);
         const auto solved = Reshape(rhs.Read(),nh,nl,n);
         mfem::forall(nl,n,[=] MFEM_HOST_DEVICE (int j, int e)
         {
            for (int i = nh-1; i >= 0; --i)
            {
               real_t v = solved(i,j,e);
               for (int k = i+1; k < nh; ++k) { v -= u(i,k)*result(k,j,first+e); }
               result(i,j,first+e) = v/u(i,i);
            }
         });
      }
   }

   // Fit all coefficients in the patch energy, fixing the stiffness null
   // space with the polynomial-exact mean before imposing any trace conditions.
   void Fit(const BatchedLOR &lor, int e, real_t volume, DenseMatrix &fit)
   {
      const int nh = Q.Width(), nl = Q.Height();
      lor.ApplyPatchStiffness(e, Q, fit);
      DenseMatrix gram(nh);
      Mult(fit, Q, gram);
      const real_t h = std::cbrt(volume);
      AddMult_a_VVt(h, mean, gram);
      AddMult_a_VWt(h, mean, weights, fit);
      DenseMatrixInverse(gram, true).Mult(fit);
      LUFactors(U.Data(), nullptr).USolve(nh, nl, fit.Data());
   }
};

// Convert canonical (increasing macrovertex ID) patch nodes to the patch
// ordering used by BatchedLOR. Each vertex permutation is built only once.
class CanonicalPatch
{
   const std::vector<BarycentricLabel> &labels;
   std::map<BarycentricLabel,int> index;
   std::map<std::array<int,4>,Array<int>> permutations;

public:
   explicit CanonicalPatch(const std::vector<BarycentricLabel> &labels_)
      : labels(labels_)
   {
      for (int i = 0; i < int(labels.size()); ++i) { index.emplace(labels[i],i); }
   }

   const Array<int> &Order(const int *vertices, const int *native_to_patch,
                           std::array<int,4> &native_to_canonical)
   {
      std::array<int,4> sorted = {0,1,2,3}, key;
      std::sort(sorted.begin(),sorted.end(),
      [&](int i, int j) { return vertices[i] < vertices[j]; });
      for (int i = 0; i < 4; ++i)
      {
         native_to_canonical[sorted[i]] = i;
         key[i] = native_to_patch[sorted[i]];
      }
      auto &order = permutations[key];
      if (order.Size()) { return order; }
      order.SetSize(labels.size());
      for (int j = 0; j < order.Size(); ++j)
      {
         BarycentricLabel a;
         for (int i = 0; i < 4; ++i) { a[key[i]] = labels[j][i]; }
         order[j] = index.at(a);
      }
      return order;
   }
};

// Each row of the DOF maps is a coarse tetrahedron in reference patch order.
class LocalTransfer : public Operator
{
   Array<int> high_dofs, lor_dofs;
   Vector inv_multiplicity;
   DenseTensor matrices;
   const bool unique_high;
   mutable Vector packed_high, packed_lor;

public:
   void ApplyReference(const Vector &x, Vector &y, bool transpose) const;
   LocalTransfer(FiniteElementSpace &high, BatchedLOR &lor,
                 TransferType type);
   MemoryClass GetMemoryClass() const override { return Device::GetMemoryClass(); }
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override;
};

void GetBoundaryMarker(const FiniteElementSpace &space, Array<int> &marker)
{
   const auto &attributes = space.GetMesh()->bdr_attributes;
   Array<int> ess_bdr(attributes.Size() ? attributes.Max() : 0);
   ess_bdr = 1;
   space.GetEssentialVDofs(ess_bdr, marker);
}

void InvertInteriorMultiplicity(Vector &multiplicity, const Array<int> &marker)
{
   const auto boundary = marker.Read();
   auto counts = multiplicity.ReadWrite();
   mfem::forall(multiplicity.Size(), [=] MFEM_HOST_DEVICE (int d)
   {
      counts[d] = boundary[d] ? 0.0 : 1.0/counts[d];
   });
}

LocalTransfer::LocalTransfer(FiniteElementSpace &high, BatchedLOR &lor,
                             TransferType type)
   : Operator(high.GetVSize(), lor.Space().GetVSize()),
     inv_multiplicity(height), unique_high(type.method == TransferMethod::Entity)
{
   Mesh &mesh = *high.GetMesh();
   Mesh &patch = lor.ReferencePatch();
   const FiniteElement &bern = *high.GetFE(0);
   const int nh = bern.GetDof(), nl = patch.GetNV(), p = bern.GetOrder();
   const bool shared = type.reference;
   std::unique_ptr<PatchBasis> basis;
   std::unique_ptr<CanonicalPatch> canonical;
   const bool gpu_blas = BatchedLinAlg::IsAvailable(BatchedLinAlg::GPU_BLAS);
   const bool device_fit = !shared && Device::Allows(Backend::DEVICE_MASK);
   matrices.SetSize(nh,nl,shared ? 1 : mesh.GetNE());
   if (unique_high)
   {
      entity::ReferenceMatrix(lor,bern,matrices(0));
      canonical = std::make_unique<CanonicalPatch>(lor.PatchLabels());
   }
   else
   {
      basis = std::make_unique<PatchBasis>(patch,bern);
      if (shared)
      { basis->Fit(lor,-1,Geometry::Volume[Geometry::TETRAHEDRON],matrices(0)); }
   }

   // Index the Bernstein coefficients by their integer barycentric exponents.
   // A permutation of the Duffy vertices induces the same permutation of these
   // exponents, including for the coefficients on shared edges and faces.
   auto alpha = BernsteinLabels(bern);
   Array3D<int> index(p+1, p+1, p+1);
   for (int i = 0; i < nh; i++)
   {
      const auto &a = alpha[i];
      index(a[1], a[2], a[3]) = i;
   }

   high_dofs.SetSize(mesh.GetNE()*nh);
   lor_dofs.SetSize(mesh.GetNE()*nl);
   if (shared && gpu_blas)
   {
      const auto mt = Device::GetDeviceMemoryType();
      packed_high.SetSize(high_dofs.Size(), mt);
      packed_lor.SetSize(lor_dofs.Size(), mt);
   }
   inv_multiplicity = 0.0;
   Array<int> hd, lor_boundary;
   GetBoundaryMarker(lor.Space(), lor_boundary);
   lor_boundary.HostRead();
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      high.GetElementDofs(e, hd);
      const int *perm = lor.VertexPermutation(e);
      std::array<int,4> sorted;
      const Array<int> *order = nullptr;
      if (canonical)
      {
         order = &canonical->Order(mesh.GetElement(e)->GetVertices(),perm,sorted);
         perm = sorted.data();
      }
      for (int i = 0; i < nh; i++)
      {
         const auto &a = alpha[i];
         const int d = hd[index(a[perm[1]], a[perm[2]], a[perm[3]])];
         // Identical entity traces need just one writer on each rank.
         high_dofs[e*nh+i] = unique_high && inv_multiplicity(d) ? -1 : d;
         inv_multiplicity(d)++;
      }

      for (int i = 0; i < nl; i++)
      {
         const int d = lor.PatchDofs()[e*nl+(order ? (*order)[i] : i)];
         lor_dofs[e*nl+i] = lor_boundary[d] ? -1 : d;
      }
      if (!shared && !device_fit)
      { basis->Fit(lor,e,lor.PatchVolumes().HostRead()[e],matrices(e)); }
   }

   if (device_fit) { basis->FitDevice(lor,matrices); }

   Array<int> high_boundary;
   GetBoundaryMarker(high, high_boundary);
   if (unique_high)
   {
      const auto boundary = high_boundary.HostRead();
      for (int i = 0; i < height; ++i)
      { inv_multiplicity(i) = boundary[i] ? 0.0 : 1.0; }
      return;
   }
   if (const Operator *P = high.GetProlongationMatrix())
   {
      Vector sums(P->Width());
      P->MultTranspose(inv_multiplicity, sums);
      P->Mult(sums, inv_multiplicity);
   }
   InvertInteriorMultiplicity(inv_multiplicity, high_boundary);
}

// Pack one column per element. The shared reference matrix acts on all columns
// in a single GEMM. Apply multiplicity weights on the high-order side only.
void LocalTransfer::ApplyReference(const Vector &x, Vector &y,
                                   bool transpose) const
{
   Vector &src = transpose ? packed_high : packed_lor;
   Vector &dst = transpose ? packed_lor : packed_high;
   const auto gather = (transpose ? high_dofs : lor_dofs).Read();
   const auto scatter = (transpose ? lor_dofs : high_dofs).Read();
   const auto weight = inv_multiplicity.Read();
   const auto input = x.Read();
   auto columns = src.Write();
   mfem::forall(src.Size(), [=] MFEM_HOST_DEVICE (int k)
   {
      const int d = gather[k];
      columns[k] = d < 0 ? 0.0 : (transpose ? weight[d]*input[d] : input[d]);
   });

   const auto &blas = BatchedLinAlg::Get(BatchedLinAlg::GPU_BLAS);
   if (transpose) { blas.MultTranspose(matrices, src, dst); }
   else { blas.Mult(matrices, src, dst); }

   const bool single_writer = unique_high && !transpose;
   y.UseDevice(true);
   if (!single_writer) { y = 0.0; }
   auto output = single_writer ? y.Write() : y.ReadWrite();
   const auto values = dst.Read();
   mfem::forall(dst.Size(), [=] MFEM_HOST_DEVICE (int k)
   {
      const int d = scatter[k];
      if (d >= 0)
      {
         const real_t v = transpose ? values[k] : weight[d]*values[k];
         if (single_writer) { output[d] = v; }
         else { AtomicAdd(output[d],v); }
      }
   });
}

void LocalTransfer::Mult(const Vector &x, Vector &y) const
{
   if (packed_high.Size()) { ApplyReference(x, y, false); return; }
   const int nh = matrices.SizeI(), nl = matrices.SizeJ();
   const auto fit = Reshape(matrices.Read(), nh, nl, matrices.SizeK());
   const bool shared = matrices.SizeK() == 1;
   const auto high = high_dofs.Read(), low = lor_dofs.Read();
   const auto weight = inv_multiplicity.Read(), input = x.Read();
   y.UseDevice(true);
   y = 0.0;
   auto output = y.ReadWrite();
   mfem::forall(high_dofs.Size(), [=] MFEM_HOST_DEVICE (int k)
   {
      if (high[k] < 0) { return; }
      const int e = k/nh, i = k%nh;
      real_t sum = 0.0;
      for (int j = 0; j < nl; j++)
      {
         const int d = low[e*nl+j];
         if (d >= 0) { sum += fit(i,j,shared ? 0 : e)*input[d]; }
      }
      AtomicAdd(output[high[k]], weight[high[k]]*sum);
   });
}

void LocalTransfer::MultTranspose(const Vector &x, Vector &y) const
{
   if (packed_high.Size()) { ApplyReference(x, y, true); return; }
   const int nh = matrices.SizeI(), nl = matrices.SizeJ();
   const auto fit = Reshape(matrices.Read(), nh, nl, matrices.SizeK());
   const bool shared = matrices.SizeK() == 1;
   const auto high = high_dofs.Read(), low = lor_dofs.Read();
   const auto weight = inv_multiplicity.Read(), input = x.Read();
   y.UseDevice(true);
   y = 0.0;
   auto output = y.ReadWrite();
   mfem::forall(lor_dofs.Size(), [=] MFEM_HOST_DEVICE (int k)
   {
      if (low[k] < 0) { return; }
      const int e = k/nl, j = k%nl;
      real_t sum = 0.0;
      for (int i = 0; i < nh; i++)
      {
         const int d = high[e*nh+i];
         if (d >= 0) { sum += fit(i,j,shared ? 0 : e)*weight[d]*input[d]; }
      }
      AtomicAdd(output[low[k]], sum);
   });
}

} // namespace

ElementTransfer::ElementTransfer(FiniteElementSpace &high,
                                 BatchedLOR &lor,
                                 TransferType type)
   : Operator(high.GetTrueVSize(), lor.Space().GetTrueVSize())
{
   const bool use_entity = type.method == TransferMethod::Entity;
   if (use_entity && !type.reference)
   {
      local = MakeEntityTransfer(high, lor);
   }
   else
   {
      local = std::make_unique<LocalTransfer>(high, lor, type);
   }
   // Select the owner's trace for either conforming entity method.
   const Operator *output = use_entity ? high.GetRestrictionTransposeOperator() :
                            high.GetProlongationMatrix();
   // Form P_out^t local P_in, eliding identities.
   const Operator *input = lor.Space().GetProlongationMatrix();
   if (IsIdentityProlongation(input) && IsIdentityProlongation(output))
   {
      transfer.Reset(local.get(), false);
   }
   else
   {
      transfer.Reset(new RAPOperator(*output, *local, *input));
   }
}

} // namespace tetlor
