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

#ifndef MFEM_MINIAPPS_TETLOR_HPP
#define MFEM_MINIAPPS_TETLOR_HPP

#include "mfem.hpp"
#include <cmath>
#include <iomanip>

namespace tetlor
{
using namespace mfem;

inline void PrintProblemSize(int p, int q, long long high_dofs,
                             long long lor_dofs)
{
   out << '\n'
       << "p = " << p << '\n'
       << "q = " << q << '\n'
       << "HO DOFs  = " << high_dofs << '\n'
       << "LOR DOFs = " << lor_dofs << "\n\n";
}

// Fit all Bernstein coefficients in the sampled LOR energy. Work in a
// mass-orthonormal basis of sampled polynomials before forming the energy Gram
// matrix: at high degree the Bernstein columns are nearly linearly dependent.
inline void EnergyFit(DenseMatrix &Q, const SparseMatrix &stiffness,
                      const Vector &lump, DenseMatrix &fit)
{
   const int nl = Q.Height(), nh = Q.Width();
   const real_t volume = lump.Sum();
   Vector root(nl);
   for (int i = 0; i < nl; i++) { root(i) = std::sqrt(lump(i)/volume); }
   DenseMatrix U(nh);
   U = 0.0;
   Vector col, prev;
   // Reorthogonalized Gram-Schmidt on diag(root) J. After unweighting the
   // columns, J = Q U and Q^t diag(lump/volume) Q = I.
   for (int j = 0; j < nh; j++)
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
   }
   fit.SetSize(nh,nl);
   Vector result(nl);
   for (int j = 0; j < nh; j++)
   {
      Q.GetColumnReference(j, col);
      col /= root;
      stiffness.Mult(col, result);
      fit.SetRow(j, result);
   }
   DenseMatrix gram(nh);
   Mult(fit, Q, gram);

   // Each Bernstein basis function has mean 1/nh. Solve U^t mean = 1/nh
   // to obtain the means of Q, and their polynomial-exact nodal quadrature.
   Vector mean(nh), weights(nl);
   for (int j = 0; j < nh; j++)
   {
      real_t value = 1.0/nh;
      for (int i = 0; i < j; i++) { value -= U(i,j)*mean(i); }
      mean(j) = value/U(j,j);
   }
   Q.Mult(mean, weights);
   weights *= lump;
   weights /= volume;
   const real_t h = std::cbrt(volume);
   AddMult_a_VVt(h, mean, gram);
   weights *= h;
   AddMultVWt(mean, weights, fit);
   DenseMatrixInverse(gram, true).Mult(fit);
   LUFactors(U.Data(), nullptr).USolve(nh, nl, fit.Data());
}

// Assemble the local transfer and the LOR stiffness from the same element matrices.
inline void AssembleTransfer(FiniteElementSpace &high, BilinearForm &low_form,
                             SparseMatrix &R)
{
   auto &low = *low_form.FESpace();
   Mesh &mesh = *high.GetMesh(), &mesh_lor = *low.GetMesh();
   const int children = mesh_lor.GetNE()/mesh.GetNE();
   const FiniteElement &bern = *high.GetFE(0);
   const int nh = bern.GetDof();
   Vector shape(nh);
   Array<int> hd, ld, micro, idx, local(mesh_lor.GetNV());
   Array<int> bdr(mesh.bdr_attributes.Max()), essential, fine_essential;
   bdr = 1;
   high.GetEssentialVDofs(bdr, essential);
   Array<int> fbdr(mesh_lor.bdr_attributes.Max());
   fbdr = 1;
   low.GetEssentialVDofs(fbdr, fine_essential);
   Vector count(high.GetVSize());
   count = 0.0;
   local = -1;

   for (int e = 0; e < mesh.GetNE(); e++)
   {
      high.GetElementDofs(e, hd);
      ld.SetSize(0);
      for (int s = e*children; s < (e+1)*children; s++)
      {
         low.GetElementDofs(s, micro);
         for (int d : micro)
         {
            if (local[d] < 0) { local[d] = ld.Size(); ld.Append(d); }
         }
      }
      const int nl = ld.Size();
      DenseMatrix sample(nl,nh);
      auto *T = mesh.GetElementTransformation(e);
      const int origin = mesh.GetElement(e)->GetVertices()[0];
      T->SetIntPoint(&Geometries.GetCenter(Geometry::TETRAHEDRON));
      Vector dx(3), xi(3);
      for (int i = 0; i < nl; i++)
      {
         for (int d = 0; d < 3; d++)
         {
            dx(d) = mesh_lor.GetVertex(ld[i])[d] - mesh.GetVertex(origin)[d];
         }
         T->InverseJacobian().Mult(dx, xi);
         IntegrationPoint ip;
         ip.Set3(xi.GetData());
         bern.CalcShape(ip, shape);
         sample.SetRow(i, shape);
      }

      SparseMatrix patch(nl);
      Vector lump(nl);
      lump = 0.0;
      for (int s = e*children; s < (e+1)*children; s++)
      {
         DenseMatrix stiffness;
         low_form.ComputeElementMatrix(s, stiffness);
         low_form.AssembleElementMatrix(s, stiffness, micro);
         idx.SetSize(micro.Size());
         for (int i = 0; i < idx.Size(); i++) { idx[i] = local[micro[i]]; }
         patch.AddSubMatrix(idx, idx, stiffness);
         const real_t weight = mesh_lor.GetElementVolume(s)/idx.Size();
         for (int i : idx) { lump(i) += weight; }
      }
      patch.Finalize();
      DenseMatrix fit;
      EnergyFit(sample, patch, lump, fit);
      R.AddSubMatrix(hd, ld, fit);
      for (int i : hd) { count(i)++; }
      for (int i : ld) { local[i] = -1; }
   }

   // The incidence count must include elements on neighboring MPI ranks.
   if (const Operator *P = high.GetProlongationMatrix())
   {
      Vector sums(P->Width());
      P->MultTranspose(count, sums);
      P->Mult(sums, count);
   }
   count.Reciprocal();
   // Enforce the exterior trace on the assembled transfer.
   for (int i = 0; i < R.Height(); i++)
   {
      if (essential[i]) { R.EliminateRow(i); }
   }
   R.EliminateCols(fine_essential);
   R.Finalize();
   R.ScaleRows(count);
}

class FictitiousSpaceSolver : public Solver
{
   const Operator &R, &aux_inv;
   const Array<int> &ess_dofs;
   mutable Vector rhs, sol;

public:
   FictitiousSpaceSolver(const Operator &R_, const Operator &aux_inv_,
                         const Array<int> &ess_dofs_)
      : Solver(R_.Height()), R(R_), aux_inv(aux_inv_),
        ess_dofs(ess_dofs_), rhs(R.Width()), sol(R.Width()) { }
   void SetOperator(const Operator &op) override { }
   void Mult(const Vector &x, Vector &y) const override
   {
      R.MultTranspose(x, rhs);
      aux_inv.Mult(rhs, sol);
      R.Mult(sol, y);
      for (int i : ess_dofs) { y(i) = x(i); }
   }
};

} // namespace tetlor
#endif
