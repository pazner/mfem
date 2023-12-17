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

#include "par_mg.hpp"

namespace mfem
{

std::vector<ParVoxelMapping> CreateParVoxelMappings(
   const int nranks,
   const int dim,
   const Array<ParentIndex> &parents,
   const Array<int> &parent_offsets,
   const Array<int> &fine_partitioning,
   const Array<int> &coarse_partitioning)
{
   std::vector<ParVoxelMapping> mappings(nranks);

   std::vector<std::vector<int>> local_coarse_elements(nranks);
   std::vector<std::unordered_map<int,int>> global_to_local_coarse(nranks);
   for (int i = 0; i < coarse_partitioning.Size(); ++i)
   {
      const int rank = coarse_partitioning[i];
      global_to_local_coarse[rank][i] = local_coarse_elements[rank].size();
      local_coarse_elements[rank].push_back(i);
   }

   std::vector<std::vector<int>> local_fine_elements(nranks);
   std::vector<std::unordered_map<int,int>> global_to_local_fine(nranks);
   for (int i = 0; i < fine_partitioning.Size(); ++i)
   {
      const int rank = fine_partitioning[i];
      global_to_local_fine[rank][i] = local_fine_elements[rank].size();
      local_fine_elements[rank].push_back(i);
   }

   const int ngrid = pow(2, dim);

   for (int i = 0; i < nranks; ++i)
   {
      const int n_local_coarse = local_coarse_elements[i].size();
      mappings[i].local_parent_offsets.SetSize(n_local_coarse + 1);
      mappings[i].local_parents.SetSize(ngrid*n_local_coarse);
   }

   std::vector<int> offsets(nranks, 0);
   std::vector<std::unordered_map<int,int>> c2f_ranks(nranks);
   std::vector<std::unordered_map<int,int>> f2c_ranks(nranks);
   for (int i = 0; i < coarse_partitioning.Size(); ++i)
   {
      const int coarse_rank = coarse_partitioning[i];
      const int local_coarse = global_to_local_coarse[coarse_rank].at(i);

      mappings[coarse_rank].local_parent_offsets[local_coarse] = offsets[coarse_rank];

      for (int j = parent_offsets[i]; j < parent_offsets[i+1]; ++j)
      {
         const auto &p = parents[j];
         const int fine_rank = fine_partitioning[p.element_index];
         const int local_fine = global_to_local_fine[fine_rank].at(p.element_index);

         if (coarse_rank == fine_rank)
         {
            const int rank = coarse_rank; // same as fine_rank
            mappings[coarse_rank].local_parents[offsets[rank]] = {local_fine, p.pmat_index};
            ++offsets[rank];
         }
         else
         {
            // Create the coarse to fine mapping
            {
               const auto result = c2f_ranks[coarse_rank].find(fine_rank);
               CoarseToFineCommunication *c2f = nullptr;
               if (result != c2f_ranks[coarse_rank].end())
               {
                  c2f = &mappings[coarse_rank].coarse_to_fine[result->second];
               }
               else
               {
                  c2f_ranks[coarse_rank][fine_rank] = mappings[coarse_rank].coarse_to_fine.size();
                  mappings[coarse_rank].coarse_to_fine.emplace_back(fine_rank);
                  c2f = &mappings[coarse_rank].coarse_to_fine.back();
               }
               c2f->coarse_to_fine.push_back({local_coarse, p.pmat_index});
            }

            // Create the fine to coarse mapping
            {
               const auto result = f2c_ranks[fine_rank].find(coarse_rank);
               FineToCoarseCommunication *f2c = nullptr;
               if (result != f2c_ranks[fine_rank].end())
               {
                  f2c = &mappings[fine_rank].fine_to_coarse[result->second];
               }
               else
               {
                  f2c_ranks[fine_rank][coarse_rank] = mappings[fine_rank].fine_to_coarse.size();
                  mappings[fine_rank].fine_to_coarse.emplace_back(coarse_rank);
                  f2c = &mappings[fine_rank].fine_to_coarse.back();
               }
               f2c->fine_to_coarse.push_back({local_fine, p.pmat_index});
            }
         }
      }
   }

   for (int i = 0; i < nranks; ++i)
   {
      mappings[i].local_parent_offsets.Last() = offsets[i];
      mappings[i].local_parents.SetSize(offsets[i]);
   }

   return mappings;
}

ParVoxelProlongation::ParVoxelProlongation(
   const ParFiniteElementSpace &coarse_fes_,
   const Array<int> &coarse_ess_dofs_,
   const ParFiniteElementSpace &fine_fes_,
   const Array<int> &fine_ess_dofs_,
   const ParVoxelMapping &mapping_)
   : Operator(fine_fes_.GetTrueVSize(), coarse_fes_.GetTrueVSize()),
     coarse_fes(coarse_fes_),
     fine_fes(fine_fes_),
     coarse_ess_dofs(coarse_ess_dofs_),
     fine_ess_dofs(fine_ess_dofs_),
     mapping(mapping_)
{
   const int dim = coarse_fes.GetMesh()->Dimension();
   const int ngrid = pow(2, dim); // number of fine elements per coarse elements

   // Set up the local prolongation matrices

   // See Mesh::UniformRefinement2D_base and Mesh::UniformRefinement3D_base
   static const double A = 0.0, B = 0.5, C = 1.0;
   // NOTE: as opposed to their definitions in the Mesh class, these are ordered
   // lexicographically
   static double quad_children[2*4*4] =
   {
      A,A, B,A, B,B, A,B, // lower-left
      B,A, C,A, C,B, B,B, // lower-right
      A,B, B,B, B,C, A,C, // upper-left
      B,B, C,B, C,C, B,C  // upper-right
   };

   static double hex_children[3*8*8] =
   {
      A,A,A, B,A,A, B,B,A, A,B,A, A,A,B, B,A,B, B,B,B, A,B,B,
      B,A,A, C,A,A, C,B,A, B,B,A, B,A,B, C,A,B, C,B,B, B,B,B,
      A,B,A, B,B,A, B,C,A, A,C,A, A,B,B, B,B,B, B,C,B, A,C,B,
      B,B,A, C,B,A, C,C,A, B,C,A, B,B,B, C,B,B, C,C,B, B,C,B,
      A,A,B, B,A,B, B,B,B, A,B,B, A,A,C, B,A,C, B,B,C, A,B,C,
      B,A,B, C,A,B, C,B,B, B,B,B, B,A,C, C,A,C, C,B,C, B,B,C,
      A,B,B, B,B,B, B,C,B, A,C,B, A,B,C, B,B,C, B,C,C, A,C,C,
      B,B,B, C,B,B, C,C,B, B,C,B, B,B,C, C,B,C, C,C,C, B,C,C,
   };

   double *ref_pmat = (dim == 2) ? quad_children : hex_children;

   const FiniteElement &fe = *coarse_fes.GetFE(0);
   const int n_loc_dof = fe.GetDof();

   IsoparametricTransformation isotr;
   isotr.SetIdentityTransformation(fe.GetGeomType());

   local_P.SetSize(n_loc_dof, n_loc_dof, ngrid);
   local_R.SetSize(n_loc_dof, n_loc_dof, ngrid);
   for (int i = 0; i < ngrid; ++i)
   {
      DenseMatrix pmat(ref_pmat + i*dim*ngrid, dim, ngrid);
      isotr.SetPointMat(pmat);
      fe.GetLocalInterpolation(isotr, local_P(i));
      fe.GetLocalRestriction(isotr, local_R(i));
   }

   u_coarse_lvec.SetSize(coarse_fes.GetVSize());
   u_fine_lvec.SetSize(fine_fes.GetVSize());
}

void ParVoxelProlongation::Mult(const Vector &u_coarse, Vector &u_fine) const
{
   const int vdim = coarse_fes.GetVDim();
   Array<int> coarse_vdofs, fine_vdofs;
   Vector u_coarse_local, u_fine_local;

   const Operator *P = coarse_fes.GetProlongationMatrix();
   if (P) { P->Mult(u_coarse, u_coarse_lvec); }
   else { u_coarse_lvec = u_coarse; }

   const int ndof_per_el = coarse_fes.GetFE(0)->GetDof();

   const int vd = 0; // <-- TODO!!!

   // Communication
   const MPI_Comm comm = coarse_fes.GetComm();
   const int nsend = mapping.coarse_to_fine.size();
   std::vector<MPI_Request> send_req(nsend);
   mult_send_buffers.resize(nsend);

   const int nrecv = mapping.fine_to_coarse.size();
   std::vector<MPI_Request> recv_req(nrecv);
   mult_recv_buffers.resize(nrecv);

   // Fill send buffers, start non-blocking send
   for (int i = 0; i < nsend; ++i)
   {
      mult_send_buffers[i].resize(mapping.coarse_to_fine[i].coarse_to_fine.size() *
                                  ndof_per_el * vdim);

      int offset = 0;
      for (const auto &c2f : mapping.coarse_to_fine[i].coarse_to_fine)
      {
         coarse_fes.GetElementDofs(c2f.coarse_element_index, coarse_vdofs);
         coarse_fes.DofsToVDofs(vd, coarse_vdofs);
         u_coarse_lvec.GetSubVector(coarse_vdofs, u_coarse_local);

         u_fine_local.NewDataAndSize(&mult_send_buffers[i][offset], ndof_per_el);

         const DenseMatrix &P = local_P(c2f.pmat_index);
         P.Mult(u_coarse_local, u_fine_local);

         offset += ndof_per_el;
      }

      MPI_Isend(mult_send_buffers[i].data(), mult_send_buffers[i].size(),
                MPI_DOUBLE, mapping.coarse_to_fine[i].rank, 0, comm, &send_req[i]);
   }

   // Allocate the receive buffers, start non-blocking receive
   for (int i = 0; i < nrecv; ++i)
   {
      mult_recv_buffers[i].resize(mapping.fine_to_coarse[i].fine_to_coarse.size() *
                                  ndof_per_el * vdim);
      MPI_Irecv(mult_recv_buffers[i].data(), mult_recv_buffers[i].size(),
                MPI_DOUBLE, mapping.fine_to_coarse[i].rank, 0, comm, &recv_req[i]);
   }

   // Local computations
   const int coarse_ne = coarse_fes.GetNE();
   for (int i = 0; i < coarse_ne; ++i)
   {
      coarse_fes.GetElementDofs(i, coarse_vdofs);
      coarse_fes.DofsToVDofs(vd, coarse_vdofs);
      u_coarse_lvec.GetSubVector(coarse_vdofs, u_coarse_local);

      for (int j = mapping.local_parent_offsets[i];
           j < mapping.local_parent_offsets[i+1]; ++j)
      {
         const ParentIndex &parent = mapping.local_parents[j];

         fine_fes.GetElementDofs(parent.element_index, fine_vdofs);
         fine_fes.DofsToVDofs(vd, fine_vdofs);
         u_fine_local.SetSize(fine_vdofs.Size());

         const DenseMatrix &P = local_P(parent.pmat_index);
         P.Mult(u_coarse_local, u_fine_local);

         u_fine_lvec.SetSubVector(fine_vdofs, u_fine_local);
      }
   }

   // Wait for receive to complete, then place received DOFs in the fine lvec
   for (int i = 0; i < nrecv; ++i)
   {
      MPI_Wait(&recv_req[i], MPI_STATUS_IGNORE);
      int offset = 0;
      for (const auto &f2c : mapping.fine_to_coarse[i].fine_to_coarse)
      {
         u_fine_local.NewDataAndSize(&mult_recv_buffers[i][offset], ndof_per_el);
         fine_fes.GetElementVDofs(f2c.fine_element_index, fine_vdofs);
         u_fine_lvec.SetSubVector(fine_vdofs, u_fine_local);
         offset += ndof_per_el;
      }
   }

   const Operator *R = fine_fes.GetRestrictionOperator();
   if (R) { R->Mult(u_fine_lvec, u_fine); }
   else { u_fine = u_fine_lvec; }

   // Essential DOFs
   for (int i : fine_ess_dofs) { u_fine[i] = 0.0; }

   // Wait for all sends to complete
   MPI_Waitall(nsend, send_req.data(), MPI_STATUSES_IGNORE);
}

void ParVoxelProlongation::MultTranspose(
   const Vector &u_fine, Vector &u_coarse) const
{
   const Operator *R = fine_fes.GetRestrictionOperator();
   if (R) { R->MultTranspose(u_fine, u_fine_lvec); }
   else { u_fine_lvec = u_fine; }

   const Operator *P = coarse_fes.GetProlongationMatrix();
   if (P) { P->MultTranspose(u_coarse_lvec, u_coarse); }

   for (int i : coarse_ess_dofs)
   {
      u_coarse[i] = 0.0;
   }
}

void ParVoxelProlongation::Coarsen(const Vector &u_fine, Vector &u_coarse) const
{
   const int vdim = coarse_fes.GetVDim();
   Array<int> coarse_vdofs, fine_vdofs;
   Vector u_coarse_local, u_fine_local;

   const Operator *P = fine_fes.GetProlongationMatrix();
   if (P) { P->Mult(u_fine, u_fine_lvec); }
   else { u_fine_lvec = u_fine; }

   const int ndof_per_el = coarse_fes.GetFE(0)->GetDof();

   const int vd = 0; // <-- TODO!!!

   // Communication
   const MPI_Comm comm = coarse_fes.GetComm();
   const int nsend = mapping.fine_to_coarse.size();
   std::vector<MPI_Request> send_req(nsend);
   mult_transp_send_buffers.resize(nsend);

   const int nrecv = mapping.coarse_to_fine.size();
   std::vector<MPI_Request> recv_req(nrecv);
   mult_transp_recv_buffers.resize(nrecv);

   // Fill send buffers, start non-blocking send
   for (int i = 0; i < nsend; ++i)
   {
      mult_transp_send_buffers[i].resize(
         mapping.fine_to_coarse[i].fine_to_coarse.size() * ndof_per_el * vdim);

      int offset = 0;
      for (const auto &f2c : mapping.fine_to_coarse[i].fine_to_coarse)
      {
         fine_fes.GetElementDofs(f2c.fine_element_index, fine_vdofs);
         fine_fes.DofsToVDofs(vd, fine_vdofs);
         u_fine_lvec.GetSubVector(fine_vdofs, u_fine_local);

         u_coarse_local.NewDataAndSize(&mult_transp_send_buffers[i][offset],
                                       ndof_per_el);

         const DenseMatrix &R = local_R(f2c.pmat_index);

         for (int k = 0; k < R.Height(); ++k)
         {
            Vector R_row(R.Width());
            R.GetRow(k, R_row);
            if (std::isfinite(R(k,0)))
            {
               u_coarse_local[k] = R_row*u_fine_local;
            }
            else
            {
               u_coarse_local[k] = std::numeric_limits<double>::infinity();
            }
         }

         offset += ndof_per_el;
      }

      MPI_Isend(mult_transp_send_buffers[i].data(),
                mult_transp_send_buffers[i].size(),
                MPI_DOUBLE, mapping.fine_to_coarse[i].rank, 0, comm, &send_req[i]);
   }

   // Allocate the receive buffers, start non-blocking receive
   for (int i = 0; i < nrecv; ++i)
   {
      mult_transp_recv_buffers[i].resize(
         mapping.coarse_to_fine[i].coarse_to_fine.size() * ndof_per_el * vdim);
      MPI_Irecv(mult_transp_recv_buffers[i].data(),
                mult_transp_recv_buffers[i].size(),
                MPI_DOUBLE, mapping.coarse_to_fine[i].rank, 0, comm, &recv_req[i]);
   }

   // Local computations
   u_coarse_lvec = 0.0;

   const int coarse_ne = coarse_fes.GetNE();
   for (int i = 0; i < coarse_ne; ++i)
   {
      coarse_fes.GetElementDofs(i, coarse_vdofs);
      coarse_fes.DofsToVDofs(vd, coarse_vdofs);

      for (int j = mapping.local_parent_offsets[i];
           j < mapping.local_parent_offsets[i+1]; ++j)
      {
         const ParentIndex &parent = mapping.local_parents[j];

         fine_fes.GetElementDofs(parent.element_index, fine_vdofs);
         fine_fes.DofsToVDofs(vd, fine_vdofs);
         u_fine_lvec.GetSubVector(fine_vdofs, u_fine_local);

         const DenseMatrix &R = local_R(parent.pmat_index);

         for (int k = 0; k < R.Height(); ++k)
         {
            if (!std::isfinite(R(k,0))) { continue; }
            Vector R_row(R.Width());
            R.GetRow(k, R_row);
            u_coarse_lvec[coarse_vdofs[k]] = R_row*u_fine_local;
         }
      }
   }

   // Wait for receive to complete, then add received DOFs to the coarse lvec
   for (int i = 0; i < nrecv; ++i)
   {
      MPI_Wait(&recv_req[i], MPI_STATUS_IGNORE);
      int offset = 0;
      for (const auto &c2f : mapping.coarse_to_fine[i].coarse_to_fine)
      {
         u_coarse_local.NewDataAndSize(&mult_transp_recv_buffers[i][offset],
                                       ndof_per_el);
         coarse_fes.GetElementVDofs(c2f.coarse_element_index, coarse_vdofs);
         for (int k = 0; k < coarse_vdofs.Size(); ++k)
         {
            const double val = u_coarse_local[k];
            if (std::isfinite(val)) { u_coarse_lvec[coarse_vdofs[k]] = val; }
         }
         offset += ndof_per_el;
      }
   }

   const Operator *R = coarse_fes.GetRestrictionOperator();
   if (R) { R->Mult(u_coarse_lvec, u_coarse); }

   // Essential DOFs
   for (int i : coarse_ess_dofs) { u_coarse[i] = 0.0; }

   // Wait for all sends to complete
   MPI_Waitall(nsend, send_req.data(), MPI_STATUSES_IGNORE);
}

}
