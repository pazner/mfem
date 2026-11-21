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
#include "lor_mesh.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

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
   void Fit(const SparseMatrix &stiffness, real_t volume, DenseMatrix &fit)
   {
      const int nh = Q.Width(), nl = Q.Height();
      fit.SetSize(nh, nl);
      Vector col, result(nl);
      for (int j = 0; j < nh; j++)
      {
         Q.GetColumnReference(j, col);
         stiffness.Mult(col, result);
         fit.SetRow(j, result);
      }
      DenseMatrix gram(nh);
      Mult(fit, Q, gram);
      const real_t h = std::cbrt(volume);
      AddMult_a_VVt(h, mean, gram);
      AddMult_a_VWt(h, mean, weights, fit);
      DenseMatrixInverse(gram, true).Mult(fit);
      LUFactors(U.Data(), nullptr).USolve(nh, nl, fit.Data());
   }
};

// The reference and physical meshes have the same child order and local
// vertex order. Express each child stiffness in reference patch numbering.
void FitPatch(PatchBasis &basis, const Mesh &patch, BilinearForm &form,
              int parent, real_t volume, bool assemble_lor, DenseMatrix &fit)
{
   SparseMatrix stiffness(patch.GetNV());
   DenseMatrix child;
   Array<int> vertices, dofs;
   for (int s = 0; s < patch.GetNE(); s++)
   {
      const int i = parent*patch.GetNE() + s;
      form.ComputeElementMatrix(i, child);
      if (assemble_lor) { form.AssembleElementMatrix(i, child, dofs); }
      patch.GetElementVertices(s, vertices);
      stiffness.AddSubMatrix(vertices, vertices, child);
   }
   stiffness.Finalize();
   basis.Fit(stiffness, volume, fit);
}

// Each row of the DOF maps is a coarse tetrahedron in reference patch order.
class LocalTransfer : public Operator
{
   Array2D<int> high_dofs, lor_dofs;
   Vector inv_multiplicity;
   std::vector<DenseMatrix> fits;

public:
   LocalTransfer(FiniteElementSpace &high, BilinearForm &lor_form,
                 int q, const std::string &grid_type, bool reference_fit);
   using Operator::SetupRAP;
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

LocalTransfer::LocalTransfer(FiniteElementSpace &high, BilinearForm &lor_form,
                             int q, const std::string &grid_type, bool reference_fit)
   : Operator(high.GetVSize(), lor_form.FESpace()->GetVSize()),
     inv_multiplicity(height), fits(reference_fit ? 1 : high.GetNE())
{
   Mesh &mesh = *high.GetMesh(), &mesh_lor = *lor_form.FESpace()->GetMesh();
   Mesh patch = MakeReferenceLORMesh(q, grid_type);
   const FiniteElement &bern = *high.GetFE(0);
   const int nh = bern.GetDof(), nl = patch.GetNV(), p = bern.GetOrder();
   PatchBasis basis(patch, bern);
   if (reference_fit)
   {
      H1_FECollection linear(1);
      FiniteElementSpace space(&patch, &linear);
      BilinearForm ref_form(&space);
      ref_form.AddDomainIntegrator(new DiffusionIntegrator);
      FitPatch(basis, patch, ref_form, 0, Geometry::Volume[Geometry::TETRAHEDRON],
               false, fits[0]);
      lor_form.Assemble();
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

   // Each reference LOR vertex needs just one child and local vertex number
   // to find its order-one H1 DOF on any physical patch.
   Array<int> child(nl), corner(nl);
   for (int s = 0; s < patch.GetNE(); s++)
   {
      const Element &el = *patch.GetElement(s);
      for (int j = 0; j < el.GetNVertices(); j++)
      {
         const int i = el.GetVertices()[j];
         child[i] = s;
         corner[i] = j;
      }
   }

   high_dofs.SetSize(mesh.GetNE(), nh);
   lor_dofs.SetSize(mesh.GetNE(), nl);
   inv_multiplicity = 0.0;
   Array<int> hd, vertices, lor_boundary;
   GetBoundaryMarker(*lor_form.FESpace(), lor_boundary);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      high.GetElementDofs(e, hd);
      GetPatchVertices(mesh, e, grid_type == "duffy", vertices);
      const int *native = mesh.GetElement(e)->GetVertices();
      int perm[4];
      for (int j = 0; j < 4; j++) { perm[j] = vertices.Find(native[j]); }
      for (int i = 0; i < nh; i++)
      {
         const int *a = alpha[i];
         const int d = hd[index(a[perm[1]], a[perm[2]], a[perm[3]])];
         high_dofs(e,i) = d;
         inv_multiplicity(d)++;
      }

      for (int i = 0; i < nl; i++)
      {
         const Element &el = *mesh_lor.GetElement(e*patch.GetNE() + child[i]);
         const int d = el.GetVertices()[corner[i]];
         lor_dofs(e,i) = lor_boundary[d] ? -1 : d;
      }
      if (!reference_fit)
      {
         FitPatch(basis, patch, lor_form, e, mesh.GetElementVolume(e), true, fits[e]);
      }
   }

   if (const Operator *P = high.GetProlongationMatrix())
   {
      Vector sums(P->Width());
      P->MultTranspose(inv_multiplicity, sums);
      P->Mult(sums, inv_multiplicity);
   }
   inv_multiplicity.Reciprocal();
   Array<int> high_boundary;
   GetBoundaryMarker(high, high_boundary);
   for (int d = 0; d < height; d++)
   {
      if (high_boundary[d]) { inv_multiplicity(d) = 0.0; }
   }
}

void LocalTransfer::Mult(const Vector &x, Vector &y) const
{
   Vector u(lor_dofs.NumCols()), v(high_dofs.NumCols());
   const real_t *input = x.HostRead();
   real_t *output = y.HostWrite();
   std::fill_n(output, height, 0.0);
   for (int e = 0; e < high_dofs.NumRows(); e++)
   {
      for (int i = 0; i < u.Size(); i++)
      {
         const int d = lor_dofs(e,i);
         u(i) = d < 0 ? 0.0 : input[d];
      }
      fits[fits.size() == 1 ? 0 : e].Mult(u, v);
      for (int i = 0; i < v.Size(); i++) { output[high_dofs(e,i)] += v(i); }
   }
   for (int i = 0; i < height; i++) { output[i] *= inv_multiplicity(i); }
}

void LocalTransfer::MultTranspose(const Vector &x, Vector &y) const
{
   Vector u(high_dofs.NumCols()), v(lor_dofs.NumCols());
   const real_t *input = x.HostRead();
   real_t *output = y.HostWrite();
   std::fill_n(output, width, 0.0);
   for (int e = 0; e < high_dofs.NumRows(); e++)
   {
      for (int i = 0; i < u.Size(); i++)
      {
         const int d = high_dofs(e,i);
         u(i) = inv_multiplicity(d)*input[d];
      }
      fits[fits.size() == 1 ? 0 : e].MultTranspose(u, v);
      for (int i = 0; i < v.Size(); i++)
      {
         const int d = lor_dofs(e,i);
         if (d >= 0) { output[d] += v(i); }
      }
   }
}

} // namespace

ElementTransfer::ElementTransfer(FiniteElementSpace &high,
                                 BilinearForm &lor_form,
                                 int q, const std::string &grid_type,
                                 bool reference_fit)
   : Operator(high.GetTrueVSize(), lor_form.FESpace()->GetTrueVSize())
{
   auto patch = std::make_unique<LocalTransfer>(high, lor_form, q, grid_type,
                                                reference_fit);
   // SetupRAP returns the local operator when both prolongations are identities.
   Operator *R = patch->SetupRAP(lor_form.FESpace()->GetProlongationMatrix(),
                                 high.GetProlongationMatrix());
   transfer.Reset(R, R != patch.get());
   local = std::move(patch);
}

} // namespace tetlor
