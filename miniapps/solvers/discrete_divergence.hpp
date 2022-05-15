// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DISCRETE_DIVERGENCE_HPP
#define MFEM_DISCRETE_DIVERGENCE_HPP

#include "mfem.hpp"
#include "general/forall.hpp"
#include "fem/lor/lor_batched.hpp" // HypreStealOwnership

namespace mfem
{

void EliminateColumns(HypreParMatrix &D, const Array<int> &ess_dofs)
{

   hypre_ParCSRMatrix *A_hypre = D;
   D.HypreReadWrite();

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A_hypre);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A_hypre);

   HYPRE_Int diag_ncols = hypre_CSRMatrixNumCols(diag);
   HYPRE_Int offd_ncols = hypre_CSRMatrixNumCols(offd);

   const int n_ess_dofs = ess_dofs.Size();

   // Start communication to figure out which columns need to be eliminated in
   // the off-diagonal block
   hypre_ParCSRCommHandle *comm_handle;
   HYPRE_Int *int_buf_data, *eliminate_col_diag, *eliminate_col_offd;
   {
      eliminate_col_diag = mfem_hypre_CTAlloc_host(HYPRE_Int, diag_ncols);
      eliminate_col_offd = mfem_hypre_CTAlloc_host(HYPRE_Int, offd_ncols);

      // Make sure A has a communication package
      hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A_hypre);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);
      }

      // Which of the local rows are to be eliminated?
      for (int i = 0; i < diag_ncols; i++)
      {
         eliminate_col_diag[i] = 0;
      }

      ess_dofs.HostRead();
      for (int i = 0; i < n_ess_dofs; i++)
      {
         eliminate_col_diag[ess_dofs[i]] = 1;
      }

      // Use a matvec communication pattern to find (in eliminate_col_offd) which of
      // the local offd columns are to be eliminated
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      int_buf_data = mfem_hypre_CTAlloc_host(
                        HYPRE_Int,
                        hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
      int index = 0;
      for (int i = 0; i < num_sends; i++)
      {
         int start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (int j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            int k = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            int_buf_data[index++] = eliminate_col_diag[k];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate(
                       11, comm_pkg, int_buf_data, eliminate_col_offd);
   }

   // Eliminate columns in the diagonal block
   {
      Memory<int> col_mem(eliminate_col_diag, offd_ncols, false);
      const auto cols = col_mem.Read(GetHypreMemoryClass(), diag_ncols);
      const int nrows_diag = hypre_CSRMatrixNumRows(diag);
      const auto I = diag->i;
      const auto J = diag->j;
      auto data = diag->data;
      MFEM_HYPRE_FORALL(i, nrows_diag,
      {
         for (int jj=I[i]; jj<I[i+1]; ++jj)
         {
            const int j = J[jj];
            data[jj] *= 1 - cols[j];
         }
      });
   }

   // Wait for MPI communication to finish
   Array<HYPRE_Int> cols_to_eliminate;
   hypre_ParCSRCommHandleDestroy(comm_handle);
   mfem_hypre_TFree_host(int_buf_data);
   mfem_hypre_TFree_host(eliminate_col_diag);

   // Eliminate columns in the off-diagonal block
   {
      Memory<int> col_mem(eliminate_col_offd, offd_ncols, false);
      const auto cols = col_mem.Read(GetHypreMemoryClass(), offd_ncols);
      const int nrows_offd = hypre_CSRMatrixNumRows(offd);
      const auto I = offd->i;
      const auto J = offd->j;
      auto data = offd->data;
      // Note: could also try a different strategy, looping over nnz in the
      // matrix and then doing a binary search in ncols_to_eliminate to see if
      // the column should be eliminated.
      MFEM_HYPRE_FORALL(i, nrows_offd,
      {
         for (int jj=I[i]; jj<I[i+1]; ++jj)
         {
            const int j = J[jj];
            data[jj] *= 1 - cols[j];
         }
      });
   }

   mfem_hypre_TFree_host(eliminate_col_offd);
}

HypreParMatrix *FormDiscreteDivergenceMatrix(ParFiniteElementSpace &fes_rt,
                                             ParFiniteElementSpace &fes_l2,
                                             const Array<int> &ess_dofs)
{
   const Mesh &mesh = *fes_rt.GetMesh();
   const int dim = mesh.Dimension();
   const int order = fes_rt.GetMaxElementOrder();

   const int n_rt = fes_rt.GetNDofs();
   const int n_l2 = fes_l2.GetNDofs();

   SparseMatrix D_local;
   D_local.OverrideSize(n_l2, n_rt);

   D_local.GetMemoryI().New(n_l2 + 1);
   // Each row always has two nonzeros
   const int nnz = n_l2*2*dim;
   auto I = D_local.HostWriteI();
   for (int i = 0; i < n_l2 + 1; ++i)
   {
      I[i] = 2*dim*i;
   }

   // element2face is a mapping of size (2*dim, nvol_per_el) such that with a
   // macro element, subelement i (in lexicographic ordering) has faces (also
   // in lexicographic order) given by the entries (j, i).

   const int o = order;
   const int op1 = order + 1;

   Array<int> element2face;
   element2face.SetSize(2*dim*pow(o, dim));

   MFEM_VERIFY(dim == 2, "Not yet 3D.");

   for (int iy = 0; iy < o; ++iy)
   {
      for (int ix = 0; ix < o; ++ix)
      {
         int ivol = ix + iy*o;
         element2face[0 + 4*ivol] = -1 - (ix + iy*op1);
         element2face[1 + 4*ivol] = ix+1 + iy*op1;
         element2face[2 + 4*ivol] = -1 - (ix + iy*o + o*op1);
         element2face[3 + 4*ivol] = ix + (iy+1)*o + o*op1;
      }
   }

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const auto *R_rt = dynamic_cast<const ElementRestriction*>(
                         fes_rt.GetElementRestriction(ordering));

   const int nel_ho = mesh.GetNE();
   const int nface_per_el = dim*order*(order+1);
   const int nvol_per_el = pow(order, dim);

   const auto gather_rt = Reshape(R_rt->gather_map.Read(), nface_per_el, nel_ho);

   const auto e2f = Reshape(element2face.Read(), 2*dim, nvol_per_el);

   // Fill J and data
   D_local.GetMemoryJ().New(nnz);
   D_local.GetMemoryData().New(nnz);

   auto J = D_local.HostWriteJ();
   auto V = D_local.HostWriteData();

   // Loop over L2 DOFs
   for (int i = 0; i < n_l2; ++i)
   {
      const int i_loc = i%nvol_per_el;
      const int i_el = i/nvol_per_el;

      for (int k = 0; k < 2*dim; ++k)
      {
         const int sjv_loc = e2f(k, i_loc);
         const int jv_loc = (sjv_loc >= 0) ? sjv_loc : -1 - sjv_loc;
         const int sgn1 = (sjv_loc >= 0) ? 1 : -1;
         MFEM_VERIFY(k < nface_per_el, "");
         const int sj = gather_rt(jv_loc, i_el);
         const int j = (sj >= 0) ? sj : -1 - sj;
         MFEM_VERIFY(j >= 0 && j < n_rt, "");
         const int sgn2 = (sj >= 0) ? 1 : -1;

         J[k + 2*dim*i] = j;
         V[k + 2*dim*i] = sgn1*sgn2;
      }
   }

   // Create a block diagonal parallel matrix
   OperatorHandle D_diag(Operator::Hypre_ParCSR);
   D_diag.MakeRectangularBlockDiag(fes_rt.GetComm(),
                                   fes_l2.GlobalVSize(),
                                   fes_rt.GlobalVSize(),
                                   fes_l2.GetDofOffsets(),
                                   fes_rt.GetDofOffsets(),
                                   &D_local);

   HypreParMatrix *D;
   // Assemble the parallel gradient matrix, must be deleted by the caller
   if (IsIdentityProlongation(fes_rt.GetProlongationMatrix()))
   {
      D = D_diag.As<HypreParMatrix>();
      D_diag.SetOperatorOwner(false);
      HypreStealOwnership(*D, D_local);
   }
   else
   {
      OperatorHandle Rt(Transpose(*fes_l2.GetRestrictionMatrix()));
      OperatorHandle Rt_diag(Operator::Hypre_ParCSR);
      Rt_diag.MakeRectangularBlockDiag(fes_l2.GetComm(),
                                       fes_l2.GlobalVSize(),
                                       fes_l2.GlobalTrueVSize(),
                                       fes_l2.GetDofOffsets(),
                                       fes_l2.GetTrueDofOffsets(),
                                       Rt.As<SparseMatrix>());
      D = RAP(Rt_diag.As<HypreParMatrix>(),
              D_diag.As<HypreParMatrix>(),
              fes_rt.Dof_TrueDof_Matrix());
   }
   D->CopyRowStarts();
   D->CopyColStarts();

   // Eliminate the boundary conditions
   EliminateColumns(*D, ess_dofs);

   return D;
}

} // namespace mfem

#endif
