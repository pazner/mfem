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
#include "voxel_integ.hpp"
#include "../../fem/picojson.h"

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
     mapping(mapping_),
     u_coarse_lvec(coarse_fes.GetVSize()),
     u_fine_lvec(fine_fes.GetVSize())
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

   const FiniteElementCollection &fec = *fine_fes.FEColl();
   const Geometry::Type geom = Geometry::TensorProductGeometry(dim);
   const FiniteElement &fe = *fec.FiniteElementForGeometry(geom);
   ndof_per_el = fe.GetDof();

   IsoparametricTransformation isotr;
   isotr.SetIdentityTransformation(fe.GetGeomType());

   local_P.SetSize(ndof_per_el, ndof_per_el, ngrid);
   local_R.SetSize(ndof_per_el, ndof_per_el, ngrid);
   for (int i = 0; i < ngrid; ++i)
   {
      DenseMatrix pmat(ref_pmat + i*dim*ngrid, dim, ngrid);
      isotr.SetPointMat(pmat);
      fe.GetLocalInterpolation(isotr, local_P(i));
      fe.GetLocalRestriction(isotr, local_R(i));
   }
}

void ParVoxelProlongation::Mult(const Vector &u_coarse, Vector &u_fine) const
{
   if (fine_fes.GetNE() == 0 && coarse_fes.GetNE() == 0) { return; }

   const int vdim = coarse_fes.GetVDim();
   Array<int> coarse_vdofs, fine_vdofs;
   Vector u_coarse_local, u_fine_local;

   const Operator *P = coarse_fes.GetProlongationMatrix();
   if (P) { P->Mult(u_coarse, u_coarse_lvec); }
   else { u_coarse_lvec = u_coarse; }

   // Communication
   const MPI_Comm comm = coarse_fes.GetComm();
   const int nsend = mapping.coarse_to_fine.size();
   std::vector<MPI_Request> send_req(nsend);
   c2f_buffers.resize(nsend);

   // Fill send buffers, start non-blocking send
   for (int i = 0; i < nsend; ++i)
   {
      c2f_buffers[i].resize(mapping.coarse_to_fine[i].coarse_to_fine.size() *
                            ndof_per_el * vdim);

      int offset = 0;

      for (int vd = 0; vd < vdim; ++vd)
      {
         for (const auto &c2f : mapping.coarse_to_fine[i].coarse_to_fine)
         {
            coarse_fes.GetElementDofs(c2f.coarse_element_index, coarse_vdofs);
            coarse_fes.DofsToVDofs(vd, coarse_vdofs);
            u_coarse_lvec.GetSubVector(coarse_vdofs, u_coarse_local);

            u_fine_local.NewDataAndSize(&c2f_buffers[i][offset], ndof_per_el);

            const DenseMatrix &P = local_P(c2f.pmat_index);
            P.Mult(u_coarse_local, u_fine_local);

            offset += ndof_per_el;
         }
      }

      MPI_Isend(c2f_buffers[i].data(), c2f_buffers[i].size(),
                MPI_DOUBLE, mapping.coarse_to_fine[i].rank, 0, comm, &send_req[i]);
   }

   // Allocate the receive buffers, start non-blocking receive
   const int nrecv = mapping.fine_to_coarse.size();
   std::vector<MPI_Request> recv_req(nrecv);
   f2c_buffers.resize(nrecv);
   for (int i = 0; i < nrecv; ++i)
   {
      f2c_buffers[i].resize(
         mapping.fine_to_coarse[i].fine_to_coarse.size() * ndof_per_el * vdim);
      MPI_Irecv(f2c_buffers[i].data(), f2c_buffers[i].size(),
                MPI_DOUBLE, mapping.fine_to_coarse[i].rank, 0, comm, &recv_req[i]);
   }

   // Local computations
   const int coarse_ne = coarse_fes.GetNE();
   // IMPORTANT: The next call to Vector::Destroy() makes sure we're not
   // pointing to the MPI send buffers anymore!
   u_fine_local.Destroy();

   for (int vd = 0; vd < vdim; ++vd)
   {
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
   }

   // Wait for receive to complete, then place received DOFs in the fine lvec
   while (true)
   {
      int i;
      MPI_Waitany(nrecv, recv_req.data(), &i, MPI_STATUS_IGNORE);
      if (i == MPI_UNDEFINED) { break; }
      int offset = 0;
      for (int vd = 0; vd < vdim; ++vd)
      {
         for (const auto &f2c : mapping.fine_to_coarse[i].fine_to_coarse)
         {
            u_fine_local.NewDataAndSize(&f2c_buffers[i][offset], ndof_per_el);
            fine_fes.GetElementDofs(f2c.fine_element_index, fine_vdofs);
            fine_fes.DofsToVDofs(vd, fine_vdofs);
            u_fine_lvec.SetSubVector(fine_vdofs, u_fine_local);
            offset += ndof_per_el;
         }
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
   if (fine_fes.GetNE() == 0 && coarse_fes.GetNE() == 0) { return; }

   const int vdim = coarse_fes.GetVDim();
   Array<int> coarse_vdofs, fine_vdofs;
   Vector u_coarse_local, u_fine_local;

   const Operator *R = fine_fes.GetRestrictionOperator();
   if (R) { R->MultTranspose(u_fine, u_fine_lvec); }
   else { u_fine_lvec = u_fine; }

   Array<bool> processed(u_fine_lvec.Size());
   processed = false;

   // Communication
   const MPI_Comm comm = coarse_fes.GetComm();
   const int nsend = mapping.fine_to_coarse.size();
   std::vector<MPI_Request> send_req(nsend);
   f2c_buffers.resize(nsend);

   // Fill send buffers, start non-blocking send
   for (int i = 0; i < nsend; ++i)
   {
      f2c_buffers[i].resize(
         mapping.fine_to_coarse[i].fine_to_coarse.size() * ndof_per_el * vdim);

      int offset = 0;
      for (int vd = 0; vd < vdim; ++vd)
      {
         for (const auto &f2c : mapping.fine_to_coarse[i].fine_to_coarse)
         {
            fine_fes.GetElementDofs(f2c.fine_element_index, fine_vdofs);
            fine_fes.DofsToVDofs(vd, fine_vdofs);
            u_fine_lvec.GetSubVector(fine_vdofs, u_fine_local);

            for (int k = 0; k < fine_vdofs.Size(); ++k)
            {
               const int k_dof = fine_vdofs[k];
               if (processed[k_dof]) { u_fine_local[k] = 0.0; }
               processed[k_dof] = true;
            }

            u_coarse_local.NewDataAndSize(&f2c_buffers[i][offset],
                                          ndof_per_el);

            const DenseMatrix &P = local_P(f2c.pmat_index);
            P.MultTranspose(u_fine_local, u_coarse_local);

            offset += ndof_per_el;
         }
      }

      MPI_Isend(f2c_buffers[i].data(), f2c_buffers[i].size(),
                MPI_DOUBLE, mapping.fine_to_coarse[i].rank, 0, comm, &send_req[i]);
   }

   // Allocate the receive buffers, start non-blocking receive
   const int nrecv = mapping.coarse_to_fine.size();
   std::vector<MPI_Request> recv_req(nrecv);
   c2f_buffers.resize(nrecv);
   for (int i = 0; i < nrecv; ++i)
   {
      c2f_buffers[i].resize(
         mapping.coarse_to_fine[i].coarse_to_fine.size() * ndof_per_el * vdim);
      MPI_Irecv(c2f_buffers[i].data(), c2f_buffers[i].size(),
                MPI_DOUBLE, mapping.coarse_to_fine[i].rank, 0, comm, &recv_req[i]);
   }

   // Local computations
   u_coarse_lvec = 0.0;
   const int coarse_ne = coarse_fes.GetNE();
   // IMPORTANT: The next call to Vector::Destroy() makes sure we're not
   // pointing to the MPI send buffers anymore!
   u_coarse_local.Destroy();

   for (int vd = 0; vd < vdim; ++vd)
   {
      for (int i = 0; i < coarse_ne; ++i)
      {
         coarse_fes.GetElementDofs(i, coarse_vdofs);
         coarse_fes.DofsToVDofs(vd, coarse_vdofs);
         u_coarse_local.SetSize(coarse_vdofs.Size());

         for (int j = mapping.local_parent_offsets[i];
              j < mapping.local_parent_offsets[i+1]; ++j)
         {
            const ParentIndex &parent = mapping.local_parents[j];

            fine_fes.GetElementDofs(parent.element_index, fine_vdofs);
            fine_fes.DofsToVDofs(vd, fine_vdofs);
            u_fine_lvec.GetSubVector(fine_vdofs, u_fine_local);

            for (int k = 0; k < fine_vdofs.Size(); ++k)
            {
               const int k_dof = fine_vdofs[k];
               if (processed[k_dof]) { u_fine_local[k] = 0.0; }
               processed[k_dof] = true;
            }

            const DenseMatrix &P = local_P(parent.pmat_index);
            P.MultTranspose(u_fine_local, u_coarse_local);

            u_coarse_lvec.AddElementVector(coarse_vdofs, u_coarse_local);
         }
      }
   }

   // Wait for receive to complete, then add received DOFs to the coarse lvec
   while (true)
   {
      int i;
      MPI_Waitany(nrecv, recv_req.data(), &i, MPI_STATUS_IGNORE);
      if (i == MPI_UNDEFINED) { break; }

      int offset = 0;

      for (int vd = 0; vd < vdim; ++vd)
      {
         for (const auto &c2f : mapping.coarse_to_fine[i].coarse_to_fine)
         {
            u_coarse_local.NewDataAndSize(&c2f_buffers[i][offset],
                                          ndof_per_el);
            coarse_fes.GetElementDofs(c2f.coarse_element_index, coarse_vdofs);
            coarse_fes.DofsToVDofs(vd, coarse_vdofs);
            u_coarse_lvec.AddElementVector(coarse_vdofs, u_coarse_local);
            offset += ndof_per_el;
         }
      }
   }

   const Operator *P = coarse_fes.GetProlongationMatrix();
   if (P) { P->MultTranspose(u_coarse_lvec, u_coarse); }
   else { u_coarse = u_coarse_lvec; }

   // Essential DOFs
   for (int i : coarse_ess_dofs) { u_coarse[i] = 0.0; }

   // Wait for all sends to complete
   MPI_Waitall(nsend, send_req.data(), MPI_STATUSES_IGNORE);
}

void ParVoxelProlongation::Coarsen(const Vector &u_fine, Vector &u_coarse) const
{
   if (fine_fes.GetNE() == 0 && coarse_fes.GetNE() == 0) { return; }

   const int vdim = coarse_fes.GetVDim();
   Array<int> coarse_vdofs, fine_vdofs;
   Vector u_coarse_local, u_fine_local;

   const Operator *P = fine_fes.GetProlongationMatrix();
   if (P) { P->Mult(u_fine, u_fine_lvec); }
   else { u_fine_lvec = u_fine; }

   // Communication
   const MPI_Comm comm = coarse_fes.GetComm();
   const int nsend = mapping.fine_to_coarse.size();
   std::vector<MPI_Request> send_req(nsend);
   f2c_buffers.resize(nsend);

   const int nrecv = mapping.coarse_to_fine.size();
   std::vector<MPI_Request> recv_req(nrecv);
   c2f_buffers.resize(nrecv);

   // Fill send buffers, start non-blocking send
   for (int i = 0; i < nsend; ++i)
   {
      f2c_buffers[i].resize(
         mapping.fine_to_coarse[i].fine_to_coarse.size() * ndof_per_el * vdim);

      int offset = 0;

      for (int vd = 0; vd < vdim; ++vd)
      {
         for (const auto &f2c : mapping.fine_to_coarse[i].fine_to_coarse)
         {
            fine_fes.GetElementDofs(f2c.fine_element_index, fine_vdofs);
            fine_fes.DofsToVDofs(vd, fine_vdofs);
            u_fine_lvec.GetSubVector(fine_vdofs, u_fine_local);

            u_coarse_local.NewDataAndSize(&f2c_buffers[i][offset],
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
      }

      MPI_Isend(f2c_buffers[i].data(),
                f2c_buffers[i].size(),
                MPI_DOUBLE, mapping.fine_to_coarse[i].rank, 0, comm, &send_req[i]);
   }

   // Allocate the receive buffers, start non-blocking receive
   for (int i = 0; i < nrecv; ++i)
   {
      c2f_buffers[i].resize(
         mapping.coarse_to_fine[i].coarse_to_fine.size() * ndof_per_el * vdim);
      MPI_Irecv(c2f_buffers[i].data(),
                c2f_buffers[i].size(),
                MPI_DOUBLE, mapping.coarse_to_fine[i].rank, 0, comm, &recv_req[i]);
   }

   // Local computations
   u_coarse_lvec = 0.0;
   const int coarse_ne = coarse_fes.GetNE();
   // IMPORTANT: The next call to Vector::Destroy() makes sure we're not
   // pointing to the MPI send buffers anymore!
   u_coarse_local.Destroy();

   for (int vd = 0; vd < vdim; ++vd)
   {
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
   }

   // Wait for receive to complete, then add received DOFs to the coarse lvec
   while (true)
   {
      int i;
      MPI_Waitany(nrecv, recv_req.data(), &i, MPI_STATUS_IGNORE);
      if (i == MPI_UNDEFINED) { break; }

      int offset = 0;

      for (int vd = 0; vd < vdim; ++vd)
      {
         for (const auto &c2f : mapping.coarse_to_fine[i].coarse_to_fine)
         {
            u_coarse_local.NewDataAndSize(&c2f_buffers[i][offset],
                                          ndof_per_el);
            coarse_fes.GetElementDofs(c2f.coarse_element_index, coarse_vdofs);
            coarse_fes.DofsToVDofs(vd, coarse_vdofs);
            for (int k = 0; k < coarse_vdofs.Size(); ++k)
            {
               const double val = u_coarse_local[k];
               if (std::isfinite(val)) { u_coarse_lvec[coarse_vdofs[k]] = val; }
            }
            offset += ndof_per_el;
         }
      }
   }

   const Operator *R = coarse_fes.GetRestrictionOperator();
   if (R) { R->Mult(u_coarse_lvec, u_coarse); }

   // Essential DOFs
   for (int i : coarse_ess_dofs) { u_coarse[i] = 0.0; }

   // Wait for all sends to complete
   MPI_Waitall(nsend, send_req.data(), MPI_STATUSES_IGNORE);
}

ParMesh LoadParMesh(const std::string &prefix)
{
   std::ifstream f(MakeParFilename(prefix + ".mesh.", Mpi::WorldRank()));
   return ParMesh(MPI_COMM_WORLD, f, false);
}

ParVoxelMapping LoadVoxelMapping(const std::string &prefix)
{
   ParVoxelMapping mapping;

   std::ifstream f(MakeParFilename(prefix + ".mapping.", Mpi::WorldRank()));
   MFEM_VERIFY(f.good(), "Error opening ifstream");

   // Load local parents
   int local_parents_size;
   f >> local_parents_size;
   mapping.local_parents.SetSize(local_parents_size);
   for (int i = 0; i < local_parents_size; ++i)
   {
      f >> mapping.local_parents[i].element_index;
      f >> mapping.local_parents[i].pmat_index;
   }

   // Load local parent offsets
   int local_parent_offsets_size;
   f >> local_parent_offsets_size;
   mapping.local_parent_offsets.SetSize(local_parent_offsets_size);
   for (int i = 0; i < local_parent_offsets_size; ++i)
   {
      f >> mapping.local_parent_offsets[i];
   }

   // Read coarse to fine
   int coarse_to_fine_size;
   f >> coarse_to_fine_size;
   mapping.coarse_to_fine.resize(coarse_to_fine_size);
   for (int i = 0; i < coarse_to_fine_size; ++i)
   {
      f >> mapping.coarse_to_fine[i].rank;
      int c2f_size;
      f >> c2f_size;
      mapping.coarse_to_fine[i].coarse_to_fine.resize(c2f_size);
      for (int j = 0; j < c2f_size; ++j)
      {
         f >> mapping.coarse_to_fine[i].coarse_to_fine[j].coarse_element_index;
         f >> mapping.coarse_to_fine[i].coarse_to_fine[j].pmat_index;
      }
   }

   // Read fine to coarse
   int fine_to_coarse_size;
   f >> fine_to_coarse_size;
   mapping.fine_to_coarse.resize(fine_to_coarse_size);
   for (int i = 0; i < fine_to_coarse_size; ++i)
   {
      f >> mapping.fine_to_coarse[i].rank;
      int f2c_size;
      f >> f2c_size;
      mapping.fine_to_coarse[i].fine_to_coarse.resize(f2c_size);
      for (int j = 0; j < f2c_size; ++j)
      {
         f >> mapping.fine_to_coarse[i].fine_to_coarse[j].fine_element_index;
         f >> mapping.fine_to_coarse[i].fine_to_coarse[j].pmat_index;
      }
   }

   return mapping;
}

ParVoxelMultigrid::ParVoxelMultigrid(const std::string &dir, int order,
                                     ProblemType pt,
                                     const std::vector<int> &ess_bdr_attrs)
{
   using namespace std;

   const int nlevels_full = [&dir]()
   {
      ifstream f(dir + "/info.json");
      picojson::value json;
      f >> json;
      const int np = json.get("np").get<double>();
      const int nlevels = json.get("nlevels").get<double>();
      MFEM_VERIFY(np == Mpi::WorldSize(), "Must run with " << np << " ranks.");
      return nlevels;
   }();

   const int coarsest_level = nlevels_full - 1;
   const int finest_level = 0;
   const int nlevels = coarsest_level - finest_level + 1;

   MFEM_VERIFY(coarsest_level < nlevels_full, "");
   MFEM_VERIFY(finest_level <= coarsest_level, "");

   if (Mpi::Root())
   {
      cout << "\nLoading hierarchy with " << nlevels << " levels.\n";
      cout << "Levels are numbered such that level 0 is the coarsest.\n";
      cout << endl;
   }

   // Load data from files. We start with the coarsest mesh, and load
   // increasingly fine meshes in sequence. (On disk, the level_0 files
   // correspond to the finest mesh).
   for (int i = coarsest_level; i >= finest_level; --i)
   {
      if (Mpi::Root()) { cout << "Loading mesh level " << meshes.size() << " (mesh " << i << ")... " << flush; }
      const string level_str = dir + "/level_" + to_string(i);
      meshes.emplace_back(new ParMesh(LoadParMesh(level_str)));
      if (i < coarsest_level)
      {
         if (Mpi::Root()) { cout << "Loading maping " << i << "... " << flush; }
         mappings.emplace_back(new ParVoxelMapping(LoadVoxelMapping(level_str)));
      }
      if (Mpi::Root()) { cout << "Done." << endl; }
   }

   if (Mpi::Root()) { cout << endl; }

   const int dim = meshes[0]->Dimension();
   fec.reset(new H1_FECollection(order, dim));

   const int vdim = [&]()
   {
      switch (pt)
      {
         case ProblemType::Poisson:
            return 1;
         case ProblemType::Elasticity:
         case ProblemType::VectorPoisson:
            return dim;
      }
   }();

   Array<int> bdr_is_ess;
   if (meshes[0]->bdr_attributes.Size() > 0)
   {
      bdr_is_ess.SetSize(meshes[0]->bdr_attributes.Max());
      bdr_is_ess = 0;
      for (int i : ess_bdr_attrs)
      {
         bdr_is_ess[i - 1] = 1;
      }
   }

   for (int i = 0; i < nlevels; ++i)
   {
      if (Mpi::Root()) { cout << "Level " << i << ":" << endl; }

      if (Mpi::Root()) { cout << "  Creating space..." << flush; }
      spaces.emplace_back(new ParFiniteElementSpace(
                             meshes[i].get(), fec.get(), vdim, Ordering::byNODES));

      HYPRE_BigInt global_ndofs = spaces[i]->GlobalTrueVSize();
      if (Mpi::Root()) { cout << "\n  Number of DOFs: " << global_ndofs; }

      ess_dofs.emplace_back(new Array<int>);
      spaces[i]->GetEssentialTrueDofs(bdr_is_ess, *ess_dofs[i]);

      const int total_ess_dofs = meshes[i]->ReduceInt(ess_dofs[i]->Size());
      if (Mpi::Root()) { cout << "\n  Number of essential DOFs: " << total_ess_dofs; }

      if (Mpi::Root()) { cout << "\n  Assembling form..." << flush; }
      forms.emplace_back(new ParBilinearForm(spaces[i].get()));
      BilinearFormIntegrator *integ = nullptr;

      switch (pt)
      {
         case ProblemType::Poisson:
            integ = new DiffusionIntegrator;
            break;
         case ProblemType::Elasticity:
            integ = new ElasticityIntegrator(lambda, mu);
            break;
         case ProblemType::VectorPoisson:
            integ = new VectorDiffusionIntegrator;
      }

      VoxelIntegrator *voxel_integ = nullptr;

      if (i > 0)
      {
         voxel_integ = new VoxelIntegrator(integ);
         forms[i]->AddDomainIntegrator(voxel_integ);
         forms[i]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         // forms[i]->AddDomainIntegrator(integ);
      }
      else
      {
         forms[i]->AddDomainIntegrator(integ);
      }
      forms[i]->Assemble();

      OperatorPtr op(Operator::ANY_TYPE);
      forms[i]->FormSystemMatrix(*ess_dofs[i], op);
      op.SetOperatorOwner(false);

      if (Mpi::Root()) { cout << "\n  Assembling diagonal..." << endl; }

      Solver *smoother;
      if (i == 0)
      {
         HypreBoomerAMG *amg = new HypreBoomerAMG(*op.As<HypreParMatrix>());
         amg->SetSystemsOptions(vdim, spaces[i]->GetOrdering() == Ordering::byNODES);
         smoother = amg;
      }
      else
      {
         // Vector diag(spaces[i]->GetTrueVSize());
         // forms[i]->AssembleDiagonal(diag);
         // smoother = new OperatorChebyshevSmoother(
         //    *op, diag, *ess_dofs[i], 2, meshes[i]->GetComm(), 30, 1e-10);
         // smoother = new OperatorJacobiSmoother(diag, *ess_dofs[i], 0.7);

         smoother = new VoxelChebyshev(*op, *spaces[i], *voxel_integ, *ess_dofs[i], 4);
      }

      AddLevel(op.Ptr(), smoother, false, true);
   }

   if (Mpi::Root()) { cout << endl; }

   for (int i = 0; i < nlevels - 1; ++i)
   {
      if (Mpi::Root()) { cout << "Prolongation level " << i << "... " << flush; }
      prolongations.emplace_back(
         new ParVoxelProlongation(*spaces[i], *ess_dofs[i], *spaces[i+1], *ess_dofs[i+1],
                                  *mappings[i]));
      if (Mpi::Root()) { cout << "Done." << endl; }
   }

   if (Mpi::Root()) { cout << endl; }
}

void ParVoxelMultigrid::FormFineLinearSystem(
   Vector &x, Vector &b, OperatorHandle &A, Vector& X, Vector& B)
{
   forms.back()->FormLinearSystem(*ess_dofs.back(), x, b, A, X, B);
}

}
