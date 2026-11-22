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

#include "tetlor.hpp"
#include <cmath>

namespace tetlor
{
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

// Each row of the DOF maps is a coarse tetrahedron in reference patch order.
class LocalTransfer : public Operator
{
   Array<int> high_dofs, lor_dofs;
   Vector inv_multiplicity;
   DenseTensor fits;

public:
   LocalTransfer(FiniteElementSpace &high, BatchedLOR &lor,
                 bool reference_fit);
   using Operator::SetupRAP;
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
                             bool reference_fit)
   : Operator(high.GetVSize(), lor.Space().GetVSize()),
     inv_multiplicity(height)
{
   Mesh &mesh = *high.GetMesh();
   Mesh &patch = lor.ReferencePatch();
   const FiniteElement &bern = *high.GetFE(0);
   const int nh = bern.GetDof(), nl = patch.GetNV(), p = bern.GetOrder();
   PatchBasis basis(patch, bern);
   fits.SetSize(nh, nl, reference_fit ? 1 : mesh.GetNE());
   if (reference_fit)
   {
      basis.Fit(lor, -1, Geometry::Volume[Geometry::TETRAHEDRON], fits(0));
   }

   // Index the Bernstein coefficients by their integer barycentric exponents.
   // A permutation of the Duffy vertices induces the same permutation of these
   // exponents, including for the coefficients on shared edges and faces.
   Array2D<int> alpha(nh, 4);
   Array3D<int> index(p+1, p+1, p+1);
   for (int i = 0; i < nh; i++)
   {
      const IntegrationPoint &ip = bern.GetNodes().IntPoint(i);
      int *a = alpha[i];
      a[1] = std::lround(p*ip.x);
      a[2] = std::lround(p*ip.y);
      a[3] = std::lround(p*ip.z);
      a[0] = p - a[1] - a[2] - a[3];
      index(a[1], a[2], a[3]) = i;
   }

   high_dofs.SetSize(mesh.GetNE()*nh);
   lor_dofs.SetSize(mesh.GetNE()*nl);
   inv_multiplicity = 0.0;
   Array<int> hd, lor_boundary;
   GetBoundaryMarker(lor.Space(), lor_boundary);
   lor_boundary.HostRead();
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      high.GetElementDofs(e, hd);
      const int *perm = lor.VertexPermutation(e);
      for (int i = 0; i < nh; i++)
      {
         const int *a = alpha[i];
         const int d = hd[index(a[perm[1]], a[perm[2]], a[perm[3]])];
         high_dofs[e*nh+i] = d;
         inv_multiplicity(d)++;
      }

      for (int i = 0; i < nl; i++)
      {
         const int d = lor.PatchDofs()[e*nl+i];
         lor_dofs[e*nl+i] = lor_boundary[d] ? -1 : d;
      }
      if (!reference_fit)
      {
         basis.Fit(lor, e, mesh.GetElementVolume(e), fits(e));
      }
   }

   if (const Operator *P = high.GetProlongationMatrix())
   {
      Vector sums(P->Width());
      P->MultTranspose(inv_multiplicity, sums);
      P->Mult(sums, inv_multiplicity);
   }
   Array<int> high_boundary;
   GetBoundaryMarker(high, high_boundary);
   InvertInteriorMultiplicity(inv_multiplicity, high_boundary);
}

void LocalTransfer::Mult(const Vector &x, Vector &y) const
{
   const int nh = fits.SizeI(), nl = fits.SizeJ();
   const auto fit = Reshape(fits.Read(), nh, nl, fits.SizeK());
   const bool shared = fits.SizeK() == 1;
   const auto high = high_dofs.Read(), low = lor_dofs.Read();
   const auto weight = inv_multiplicity.Read(), input = x.Read();
   y.UseDevice(true);
   y = 0.0;
   auto output = y.ReadWrite();
   mfem::forall(high_dofs.Size(), [=] MFEM_HOST_DEVICE (int k)
   {
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
   const int nh = fits.SizeI(), nl = fits.SizeJ();
   const auto fit = Reshape(fits.Read(), nh, nl, fits.SizeK());
   const bool shared = fits.SizeK() == 1;
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
         sum += fit(i,j,shared ? 0 : e)*weight[d]*input[d];
      }
      AtomicAdd(output[low[k]], sum);
   });
}

} // namespace

ElementTransfer::ElementTransfer(FiniteElementSpace &high,
                                 BatchedLOR &lor,
                                 bool reference_fit)
   : Operator(high.GetTrueVSize(), lor.Space().GetTrueVSize())
{
   auto patch = std::make_unique<LocalTransfer>(high, lor, reference_fit);
   // SetupRAP returns the local operator when both prolongations are identities.
   Operator *R = patch->SetupRAP(lor.Space().GetProlongationMatrix(),
                                 high.GetProlongationMatrix());
   transfer.Reset(R, R != patch.get());
   local = std::move(patch);
}

} // namespace tetlor
