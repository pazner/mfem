// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "hybridization_ext.hpp"
#include "hybridization.hpp"
#include "../general/forall.hpp"

namespace mfem
{

HybridizationExtension::HybridizationExtension(Hybridization &hybridization_)
   : h(hybridization_)
{ }

void HybridizationExtension::ConstructC()
{
   Mesh &mesh = *h.fes.GetMesh();
   const int dim = mesh.Dimension();
   const int ne = mesh.GetNE();

   const int n_hat_dof_per_el = h.fes.GetFE(0)->GetDof();
   const int n_c_dof_per_face = h.c_fes.GetFaceElement(0)->GetDof();
   const int n_faces_per_el = [&mesh, dim]()
   {
      switch (dim)
      {
         case 2: return mesh.GetElement(0)->GetNEdges();
         case 3: return mesh.GetElement(0)->GetNFaces();
         default: return -1;
      }
   }();

   el_to_face.SetSize(ne * n_faces_per_el);
   Ct_mat.SetSize(ne * n_hat_dof_per_el * n_c_dof_per_face * n_faces_per_el);
   Ct_mat.UseDevice(true);
   Ct_mat = 0.0;

   el_to_face = -1;
   int face_idx = 0;

   for (int f = 0; f < mesh.GetNumFaces(); ++f)
   {
      const Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      if (!info.IsInterior()) { continue; }

      FaceElementTransformations *FTr = mesh.GetInteriorFaceTransformations(f);
      MFEM_ASSERT(FTr, "Invalid interior face.");

      const int el1 = info.element[0].index;
      const int fi1 = info.element[0].local_face_id;
      el_to_face[el1 * n_faces_per_el + fi1] = face_idx;

      const int el2 = info.element[1].index;
      const int fi2 = info.element[1].local_face_id;
      el_to_face[el2 * n_faces_per_el + fi2] = face_idx;


      DenseMatrix elmat;
      h.c_bfi->AssembleFaceMatrix(*h.c_fes.GetFaceElement(f),
                                  *h.fes.GetFE(el1),
                                  *h.fes.GetFE(el2),
                                  *FTr, elmat);
      elmat.Threshold(1e-12 * elmat.MaxMaxNorm());

      const int sz = n_hat_dof_per_el * n_c_dof_per_face;
      MFEM_ASSERT(2*sz == elmat.Width()*elmat.Height(), "");

      const int offset1 = (el1*n_faces_per_el + fi1)*sz;
      const int offset2 = (el2*n_faces_per_el + fi2)*sz;
      for (int j = 0; j < n_c_dof_per_face; ++j)
      {
         for (int i = 0; i < n_hat_dof_per_el; ++i)
         {
            // std::copy(elmat.GetData(), elmat.GetData() + sz, &Ct_mat[offset1]);
            // std::copy(elmat.GetData() + sz, elmat.GetData() + 2*sz, &Ct_mat[offset2]);

            Ct_mat[offset1 + i + j*n_hat_dof_per_el] += elmat(i, j);
            Ct_mat[offset2 + i + j*n_hat_dof_per_el] += elmat(n_hat_dof_per_el + i, j);
         }
      }

      ++face_idx;
   }
}

void HybridizationExtension::MultCt(const Vector &x, Vector &y) const
{
   Mesh &mesh = *h.fes.GetMesh();
   const int dim = mesh.Dimension();
   const int ne = mesh.GetNE();

   const int n_hat_dof_per_el = h.fes.GetFE(0)->GetDof();
   const int n_c_dof_per_face = h.c_fes.GetFaceElement(0)->GetDof();
   const int n_faces_per_el = [&mesh, dim]()
   {
      switch (dim)
      {
         case 2: return mesh.GetElement(0)->GetNEdges();
         case 3: return mesh.GetElement(0)->GetNFaces();
         default: return -1;
      }
   }();

   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   const FaceRestriction *face_restr = h.c_fes.GetFaceRestriction(
                                          ordering, FaceType::Interior);

   Vector x_evec(face_restr->Height());
   face_restr->Mult(x, x_evec);

   // Size of the element matrix;
   const int sz = n_hat_dof_per_el * n_c_dof_per_face;

   y = 0.0;
   for (int e = 0; e < ne; ++e)
   {
      for (int fi = 0; fi < n_faces_per_el; ++fi)
      {
         const int f = el_to_face[e*n_faces_per_el + fi];
         if (f < 0) { continue; }
         const int offset = (e*n_faces_per_el + fi) * sz;

         for (int j = 0; j < n_c_dof_per_face; ++j)
         {
            for (int i = 0; i < n_hat_dof_per_el; ++i)
            {
               y[e*n_hat_dof_per_el + i] +=
                  Ct_mat[offset + i + j*n_hat_dof_per_el]*x_evec[j + f*n_c_dof_per_face];
            }
         }
      }
   }
}

void HybridizationExtension::Init(const Array<int> &ess_tdof_list)
{
   // Should verify that the preconditions are met:
   // - No variable p
   // - All elements to same (tensor-product?)
   // - Dimension = 2 or 3

   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;

   const Operator *R_op = h.fes.GetElementRestriction(ordering);
   const auto *R = dynamic_cast<const ElementRestriction*>(R_op);
   MFEM_VERIFY(R, "");

   const int ne = h.fes.GetNE();

   // count the number of dofs in the discontinuous version of fes:
   const int ndof_per_el = h.fes.GetFE(0)->GetDof();
   num_hat_dofs = ne*ndof_per_el;
   {
      h.hat_offsets.SetSize(ne + 1);
      int *d_hat_offsets = h.hat_offsets.Write();
      mfem::forall(ne + 1, [=] MFEM_HOST_DEVICE (int i)
      {
         d_hat_offsets[i] = i*ndof_per_el;
      });
   }

   ConstructC();
   h.ConstructC();

   // We now split the "hat DOFs" (broken DOFs) into three classes of DOFs, each
   // marked with an integer:
   //
   //  1: essential DOFs (depending only on essential Lagrange multiplier DOFs)
   //  0: free interior DOFs (the corresponding column in the C matrix is zero,
   //     this happens when the DOF is in the interior on an element)
   // -1: free boundary DOFs (free DOFs that lie on the interface between two
   //     elements)
   //
   // These integers are used to define the values in HatDofType enum.
   {
      const int ntdofs = h.fes.GetTrueVSize();
      // free_tdof_marker is 1 if the DOF is free, 0 if the DOF is essential
      Array<int> free_tdof_marker(ntdofs);
      {
         int *d_free_tdof_marker = free_tdof_marker.Write();
         mfem::forall(ntdofs, [=] MFEM_HOST_DEVICE (int i)
         {
            d_free_tdof_marker[i] = 1;
         });
         const int n_ess_dofs = ess_tdof_list.Size();
         const int *d_ess_tdof_list = ess_tdof_list.Read();
         mfem::forall(n_ess_dofs, [=] MFEM_HOST_DEVICE (int i)
         {
            d_free_tdof_marker[d_ess_tdof_list[i]] = 0;
         });
      }

      Array<int> free_vdofs_marker;
      const SparseMatrix *cP = h.fes.GetConformingProlongation();
      if (!cP)
      {
         free_vdofs_marker.MakeRef(free_tdof_marker);
      }
      else
      {
         free_vdofs_marker.SetSize(h.fes.GetVSize());
         cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
      }

      h.hat_dofs_marker.SetSize(num_hat_dofs);
      {
         // The gather map from the ElementRestriction operator gives us the
         // index of the L-dof corresponding to a given (element, local DOF)
         // index pair.
         const int *gather_map = R->GatherMap().Read();
         const int *d_free_vdofs_marker = free_vdofs_marker.Read();
         int *d_hat_dofs_marker = h.hat_dofs_marker.Write();

         // Set the hat_dofs_marker to 1 or 0 according to whether the DOF is
         // "free" or "essential". (For now, we mark all free DOFs as free
         // interior as a placeholder). Then, as a later step, the "free" DOFs
         // will be further classified as "interior free" or "boundary free".
         mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
         {
            const int j_s = gather_map[i];
            const int j = (j_s >= 0) ? j_s : -1 - j_s;
            d_hat_dofs_marker[i] = d_free_vdofs_marker[j] ? FREE_INTERIOR : ESSENTIAL;
         });

         // A free DOF is "interior" or "internal" if the corresponding column
         // of C (row of C^T) is zero. We mark the free DOFs with non-zero
         // columns of C as boundary free DOFs.
         const int *d_I = h.Ct->ReadI();
         mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
         {
            const int row_size = d_I[i+1] - d_I[i];
            if (d_hat_dofs_marker[i] == 0 && row_size > 0)
            {
               d_hat_dofs_marker[i] = FREE_BOUNDARY;
            }
         });
      }
   }

   // Define Af_offsets and Af_f_offsets. Af_offsets are the offsets of the
   // matrix blocks into the data array Af_data.
   //
   // Af_f_offsets are the offets of the pivots (coming from LU factorization)
   // that are stored in the data array Af_ipiv.
   //
   // NOTE: as opposed to the non-device version of hybridization, the essential
   // DOFs are included in these matrices to ensure that all matrices have
   // identical sizes. This enabled efficient batched matrix computations.

   h.Af_offsets.SetSize(ne+1);
   h.Af_f_offsets.SetSize(ne+1);
   {
      int *Af_offsets = h.Af_offsets.Write();
      int *Af_f_offsets = h.Af_f_offsets.Write();
      mfem::forall(ne + 1, [=] MFEM_HOST_DEVICE (int i)
      {
         Af_f_offsets[i] = i*ndof_per_el;
         Af_offsets[i] = i*ndof_per_el*ndof_per_el;
      });
   }

   h.Af_data.SetSize(ne*ndof_per_el*ndof_per_el);
   h.Af_ipiv.SetSize(ne*ndof_per_el);
}

void HybridizationExtension::ComputeSolution(
   const Vector &b, const Vector &sol_r, Vector &sol) const
{
   // First, represent the solution on the "broken" space with vector 'tmp1'.

   // tmp1 = Af^{-1} ( Rf^t - Cf^t sol_r )
   tmp1.SetSize(num_hat_dofs);
   h.MultAfInv(b, sol_r, tmp1, 1);

   // The T-vector 'sol' has the correct essential boundary conditions. We
   // covert sol to an L-vector ('tmp2') to preserve those boundary values in
   // the output solution.
   const Operator *R = h.fes.GetRestrictionOperator();
   if (!R)
   {
      MFEM_ASSERT(sol.Size() == h.fes.GetVSize(), "");
      tmp2.MakeRef(sol, 0);
   }
   else
   {
      tmp2.SetSize(R->Width());
      R->MultTranspose(sol, tmp2);
   }

   // Move vector 'tmp1' from broken space to L-vector 'tmp2'.
   {
      const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
      const auto *R = static_cast<const ElementRestriction*>(
                         h.fes.GetElementRestriction(ordering));
      const int *gather_map = R->GatherMap().Read();
      const int *d_hat_dofs_marker = h.hat_dofs_marker.Read();
      const double *d_evec = tmp1.ReadWrite();
      double *d_lvec = tmp2.ReadWrite();
      mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
      {
         // Skip essential DOFs
         if (d_hat_dofs_marker[i] == ESSENTIAL) { return; }

         const int j_s = gather_map[i];
         const int sgn = (j_s >= 0) ? 1 : -1;
         const int j = (j_s >= 0) ? j_s : -1 - j_s;

         d_lvec[j] = sgn*d_evec[i];
      });
   }

   // Finally, convert from L-vector to T-vector.
   if (R)
   {
      R->Mult(tmp2, sol);
   }
}

}