// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "patch_multigrid.hpp"

using namespace std;

namespace mfem
{

PatchSmoother::PatchSmoother(const FiniteElementSpace &fes,
                             const SparseMatrix &A,
                             const Array<int> &ess_dofs_,
                             real_t alpha_)
   : Solver(fes.GetTrueVSize()), ess_dofs(ess_dofs_), alpha(alpha_)
{
   unique_ptr<Table> dof2el;
   {
      Table elem_dof_table = fes.GetElementToDofTable(); // deep copy
      {
         const int nnz = elem_dof_table.GetI()[elem_dof_table.Size()];
         int *J = elem_dof_table.GetJ();
         for (int i = 0; i < nnz; ++i)
         {
            const int j = J[i];
            J[i] = (j >= 0) ? j : -1 - j;
         }
      }
      dof2el.reset(Transpose(elem_dof_table));
   }

   set<int> ess_dof_set(ess_dofs.begin(), ess_dofs.end());

   unique_ptr<Table> v2el(fes.GetMesh()->GetVertexToElementTable());
   patch_dofs.resize(fes.GetNV());
   Array<int> el_dofs, row;

   for (int iv = 0; iv < fes.GetNV(); ++iv)
   {
      Array<int> &dofs = patch_dofs[iv];
      v2el->GetRow(iv, row);

      set<int> patch_el_set(row.begin(), row.end());
      for (const int iel : row)
      {
         fes.GetElementVDofs(iel, el_dofs);
         for (const int s_dof : el_dofs)
         {
            const int i_dof = (s_dof >= 0) ? s_dof : -1 - s_dof;

            Array<int> dof_els;
            dof2el->GetRow(i_dof, dof_els);
            bool bdr_dof = false;
            for (const int iel2 : dof_els)
            {
               if (patch_el_set.find(iel2) == patch_el_set.end())
               {
                  // This dof belongs to a non-patch element. It is on the
                  // boundary of the patch.
                  bdr_dof = true;
               }
            }
            if (ess_dof_set.find(i_dof) != ess_dof_set.end())
            {
               bdr_dof = true;
            }

            if (!bdr_dof) { dofs.Append(i_dof); }
         }
      }
      dofs.Sort();
      dofs.Unique();

      DenseMatrix submatrix(dofs.Size(), dofs.Size());
      A.GetSubMatrix(dofs, dofs, submatrix);
      DenseMatrixInverse submatrix_inv(submatrix);

      DenseMatrix *patch_solv = new DenseMatrix(dofs.Size(), dofs.Size());
      submatrix_inv.GetInverseMatrix(*patch_solv);

      patches_inv.emplace_back(patch_solv);
   }
}

void PatchSmoother::SetOperator(const Operator &op)
{
   MFEM_ABORT("Not supported.");
}

void PatchSmoother::Mult(const Vector &b, Vector &x) const
{
   x = 0.0;
   for (int i = 0; i < patches_inv.size(); ++i)
   {
      const Array<int> &dofs = patch_dofs[i];
      if (dofs.Size() > 0)
      {
         z1.SetSize(dofs.Size());
         z2.SetSize(dofs.Size());

         b.GetSubVector(dofs, z1);
         patches_inv[i]->Mult(z1, z2);
         z2 *= alpha / 3.0;
         x.AddElementVector(dofs, z2);
      }
   }
   for (const int i : ess_dofs)
   {
      x[i] = b[i];
   }
}

void PatchSmoother::MultTranspose(const Vector &b, Vector &x) const
{
   Mult(b, x);
}

PatchMultigrid::PatchMultigrid(FiniteElementSpaceHierarchy &fespaces,
                               Array<int> &ess_bdr,
                               real_t lambda_)
   : GeometricMultigrid(fespaces, ess_bdr),
     lambda(lambda_)
{
   for (int level = 0; level < fespaces.GetNumLevels(); ++level)
   {
      FormLevel(fespaces.GetFESpaceAtLevel(level), level);
   }
}

void PatchMultigrid::FormLevel(FiniteElementSpace &fes, int level)
{
   // Create the bilinear form
   BilinearForm *a = new BilinearForm(&fes);

   const int order = fes.GetMaxElementOrder();
   const real_t kappa = (order + 1)*(order + 1);


   a->AddDomainIntegrator(new VectorFEDiffusionIntegrator);
   a->AddDomainIntegrator(new VectorFEMassIntegrator);
   a->AddDomainIntegrator(new DivDivIntegrator(lambda));
   a->AddInteriorFaceIntegrator(new VectorFE_DGDiffusionIntegrator(kappa));
   a->AddBdrFaceIntegrator(new VectorFE_DGDiffusionIntegrator(kappa));

   a->SetDiagonalPolicy(Operator::DIAG_ONE);

   a->Assemble();
   bfs.Append(a);

   // Form the operator
   OperatorHandle A;
   const Array<int> &ess_dofs = *essentialTrueDofs[level];
   a->FormSystemMatrix(ess_dofs, A);
   const bool own_op = A.OwnsOperator();
   A.SetOperatorOwner(false);
   SparseMatrix *A_mat = A.As<SparseMatrix>();

   Solver *S;
   if (level == 0)
   {
      S = new UMFPackSolver(*A_mat);
   }
   else
   {
      S = new PatchSmoother(fes, *A_mat, ess_dofs);
   }

   AddLevel(A.Ptr(), S, own_op, true);
}

} // namespace mfem
