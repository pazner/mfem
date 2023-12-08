// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mg.hpp"

namespace mfem
{

ImageMesh &GetImageMesh(const FiniteElementSpace &fes)
{
   auto *mesh_ptr = dynamic_cast<ImageMesh*>(fes.GetMesh());
   MFEM_VERIFY(mesh_ptr != nullptr, "Mesh must be ImageMesh");
   return *mesh_ptr;
}

ImageProlongation::ImageProlongation(
   const FiniteElementSpace &coarse_fes_,
   const Array<int> &coarse_ess_dofs_,
   const FiniteElementSpace &fine_fes_,
   const Array<int> &fine_ess_dofs_)
   : Operator(fine_fes_.GetTrueVSize(), coarse_fes_.GetTrueVSize()),
     coarse_fes(coarse_fes_),
     fine_fes(fine_fes_),
     coarse_ess_dofs(coarse_ess_dofs_),
     fine_ess_dofs(fine_ess_dofs_)
{
   ImageMesh &coarse_mesh = GetImageMesh(coarse_fes);
   ImageMesh &fine_mesh = GetImageMesh(fine_fes);

   const int coarse_ne = coarse_mesh.GetNE();

   parent_offsets.SetSize(coarse_ne + 1);
   parents.SetSize(4*coarse_ne);

   int offset = 0;
   for (int i = 0; i < coarse_ne; ++i)
   {
      parent_offsets[i] = offset;
      const LexIndex lex = coarse_mesh.GetLexicographicIndex(i);

      for (int jj = 0; jj < 2; ++jj)
      {
         for (int ii = 0; ii < 2; ++ii)
         {
            const int fine_idx = fine_mesh.GetElementIndex(2*lex[0] + ii, 2*lex[1] + jj);
            if (fine_idx >= 0)
            {
               parents[offset] = {fine_idx, ii + 2*jj};
               ++offset;
            }
         }
      }
   }
   parent_offsets.Last() = offset;
   parents.SetSize(offset);

   // Set up the local prolongation matrices

   // See Mesh::UniformRefinement2D_base
   static const double A = 0.0, B = 0.5, C = 1.0;
   // NOTE: as opposed to Mesh::UniformRefinement2D_base, these are ordered
   // lexicographically
   static double quad_children[2*4*4] =
   {
      A,A, B,A, B,B, A,B, // lower-left
      B,A, C,A, C,B, B,B, // lower-right
      A,B, B,B, B,C, A,C,  // upper-left
      B,B, C,B, C,C, B,C // upper-right
   };

   const FiniteElement &fe = *coarse_fes.GetFE(0);
   const int n_loc_dof = fe.GetDof();

   IsoparametricTransformation isotr;
   isotr.SetIdentityTransformation(fe.GetGeomType());

   local_P.SetSize(n_loc_dof, n_loc_dof, 4);
   local_R.SetSize(n_loc_dof, n_loc_dof, 4);
   for (int i = 0; i < 4; ++i)
   {
      DenseMatrix pmat(quad_children + i*2*4, 2, 4);
      isotr.SetPointMat(pmat);
      fe.GetLocalInterpolation(isotr, local_P(i));
      fe.GetLocalRestriction(isotr, local_R(i));
   }
}

void ImageProlongation::Mult(const Vector &u_coarse, Vector &u_fine) const
{
   const int coarse_ne = coarse_fes.GetNE();
   const int vdim = coarse_fes.GetVDim();
   Array<int> coarse_vdofs, fine_vdofs;
   Vector u_coarse_local, u_fine_local;
   for (int vd = 0; vd < vdim; ++vd)
   {
      for (int i = 0; i < coarse_ne; ++i)
      {
         coarse_fes.GetElementDofs(i, coarse_vdofs);
         coarse_fes.DofsToVDofs(vd, coarse_vdofs);
         u_coarse.GetSubVector(coarse_vdofs, u_coarse_local);

         for (int j = parent_offsets[i]; j < parent_offsets[i+1]; ++j)
         {
            const ParentIndex &parent = parents[j];

            fine_fes.GetElementDofs(parent.element_index, fine_vdofs);
            fine_fes.DofsToVDofs(vd, fine_vdofs);
            u_fine_local.SetSize(fine_vdofs.Size());

            const DenseMatrix &P = local_P(parent.pmat_index);
            P.Mult(u_coarse_local, u_fine_local);

            u_fine.SetSubVector(fine_vdofs, u_fine_local);
         }
      }
   }

   for (int i : fine_ess_dofs)
   {
      u_fine[i] = 0.0;
   }
}

void ImageProlongation::MultTranspose(
   const Vector &u_fine, Vector &u_coarse) const
{
   u_coarse = 0.0;

   Array<bool> processed(u_fine.Size());
   processed = false;

   const int coarse_ne = coarse_fes.GetNE();
   const int vdim = coarse_fes.GetVDim();

   Array<int> coarse_vdofs, fine_vdofs;
   Vector u_coarse_local, u_fine_local;

   for (int vd = 0; vd < vdim; ++vd)
   {
      for (int i = 0; i < coarse_ne; ++i)
      {
         coarse_fes.GetElementDofs(i, coarse_vdofs);
         coarse_fes.DofsToVDofs(vd, coarse_vdofs);
         u_coarse_local.SetSize(coarse_vdofs.Size());

         for (int j = parent_offsets[i]; j < parent_offsets[i+1]; ++j)
         {
            const ParentIndex &parent = parents[j];

            fine_fes.GetElementDofs(parent.element_index, fine_vdofs);
            fine_fes.DofsToVDofs(vd, fine_vdofs);
            u_fine.GetSubVector(fine_vdofs, u_fine_local);

            for (int k = 0; k < fine_vdofs.Size(); ++k)
            {
               if (processed[fine_vdofs[k]]) { u_fine_local[k] = 0.0; }
            }

            const DenseMatrix &P = local_P(parent.pmat_index);
            P.MultTranspose(u_fine_local, u_coarse_local);

            u_coarse.AddElementVector(coarse_vdofs, u_coarse_local);

            for (int k : fine_vdofs)
            {
               processed[k] = true;
            }
         }
      }
   }

   for (int i : coarse_ess_dofs)
   {
      u_coarse[i] = 0.0;
   }
}

void ImageProlongation::Coarsen(const Vector &u_fine, Vector &u_coarse) const
{
   u_coarse = 0.0;

   const int coarse_ne = coarse_fes.GetNE();
   const int vdim = coarse_fes.GetVDim();

   Array<int> coarse_vdofs, fine_vdofs;
   Vector u_coarse_local, u_fine_local;

   for (int vd = 0; vd < vdim; ++vd)
   {
      for (int i = 0; i < coarse_ne; ++i)
      {
         coarse_fes.GetElementDofs(i, coarse_vdofs);
         coarse_fes.DofsToVDofs(vd, coarse_vdofs);
         u_coarse_local.SetSize(coarse_vdofs.Size());

         for (int j = parent_offsets[i]; j < parent_offsets[i+1]; ++j)
         {
            const ParentIndex &parent = parents[j];

            fine_fes.GetElementDofs(parent.element_index, fine_vdofs);
            fine_fes.DofsToVDofs(vd, fine_vdofs);
            u_fine.GetSubVector(fine_vdofs, u_fine_local);

            const DenseMatrix &R = local_R(parent.pmat_index);

            for (int k = 0; k < R.Height(); ++k)
            {
               if (!std::isfinite(R(k,0))) { continue; }
               Vector R_row(R.Width());
               R.GetRow(k, R_row);
               u_coarse[coarse_vdofs[k]] = R_row*u_fine_local;
            }
         }
      }
   }

   for (int i : coarse_ess_dofs)
   {
      u_coarse[i] = 0.0;
   }
}

ImageMultigrid::ImageMultigrid(const ImageMesh &&fine_mesh,
                               FiniteElementCollection &fec)
{
   meshes.emplace_back(new ImageMesh(fine_mesh));
   ImageMesh *current_mesh = meshes.back().get();

   while (current_mesh->GetImage().Width() >= 4 &&
          current_mesh->GetImage().Height() >= 4 &&
          !current_mesh->GetImage().Coarsen().Empty())
   {
      std::cout << "Current mesh: " << current_mesh->GetNE() << '\n';
      ImageMesh *new_mesh = new ImageMesh(current_mesh->Coarsen());
      std::cout << "New mesh:     " << new_mesh->GetNE() << '\n';
      meshes.emplace_back(new_mesh);
      current_mesh = new_mesh;
   }

   std::reverse(meshes.begin(), meshes.end());

   nlevels = meshes.size();

   for (int i = 0; i < nlevels; ++i)
   {
      spaces.emplace_back(new FiniteElementSpace(meshes[i].get(), &fec));

      ess_dofs.emplace_back(new Array<int>);
      spaces[i]->GetBoundaryTrueDofs(*ess_dofs[i]);

      forms.emplace_back(new BilinearForm(spaces[i].get()));
      forms[i]->AddDomainIntegrator(new DiffusionIntegrator);
      // forms[i]->AddDomainIntegrator(new MassIntegrator);
      forms[i]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      forms[i]->Assemble();

      OperatorPtr opr(Operator::ANY_TYPE);
      forms[i]->FormSystemMatrix(*ess_dofs[i], opr);
      opr.SetOperatorOwner(false);

      Vector diag(spaces[i]->GetTrueVSize());
      forms[i]->AssembleDiagonal(diag);

      Solver *smoother = new OperatorChebyshevSmoother(
         *opr, diag, *ess_dofs[i], 2);
      AddLevel(opr.Ptr(), smoother, true, true);
   }

   for (int i = 0; i < nlevels - 1; ++i)
   {
      prolongations.emplace_back(
         new ImageProlongation(*spaces[i], *ess_dofs[i], *spaces[i+1], *ess_dofs[i+1]));
   }
}

void ImageMultigrid::FormFineLinearSystem(
   Vector &x, Vector &b, OperatorHandle &A, Vector& X, Vector& B)
{
   forms.back()->FormLinearSystem(*ess_dofs.back(), x, b, A, X, B);
}

}
