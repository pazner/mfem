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

#ifndef MFEM_HDIV_HB_HPP
#define MFEM_HDIV_HB_HPP

namespace mfem
{

inline real_t Sign(int d) { return d < 0 ? -1.0 : 1.0; }

struct BrokenSpace
{
   ParFiniteElementSpace &fes;
   // The DG interface space supplies face numbering, not a weak trace basis:
   // its coefficients label the existing IH flux-dual multiplier coordinates.
   DG_Interface_FECollection trace_fec;
   ParFiniteElementSpace trace_fes;
   const ElementRestriction &restriction;
   const Operator &prolongation;
   const int nd;
   const Array<int> &dofs;
   HYPRE_BigInt num_global_flux_dofs, num_global_trace_dofs;
   Array<int> multiplicity, trace_local, inactive_trace;
   Array<HYPRE_BigInt> trace_global;
   const HypreParMatrix &trace_prolongation;
   std::unique_ptr<HypreParMatrix> Ct;
   std::unique_ptr<TransposeOperator> C;

   BrokenSpace(ParFiniteElementSpace &fes_)
      : fes(fes_), trace_fec(fes.GetMaxElementOrder()-1, fes.GetMesh()->Dimension()),
        trace_fes(fes.GetParMesh(), &trace_fec),
        restriction(dynamic_cast<const ElementRestriction&>(
                       *fes.GetElementRestriction(ElementDofOrdering::NATIVE))),
        prolongation(*fes.GetProlongationMatrix()),
        nd(fes.GetTypicalFE()->GetDof()), dofs(restriction.GatherMap()),
        multiplicity(fes.GetVSize()), trace_local(trace_fes.GetVSize()),
        trace_global(trace_fes.GetVSize()),
        trace_prolongation(*trace_fes.Dof_TrueDof_Matrix())
   {
      const MPI_Comm comm = fes.GetComm();
      const Array<int> &offsets = restriction.Offsets();
      for (int d = 0; d < fes.GetVSize(); ++d)
      {
         multiplicity[d] = offsets[d+1]-offsets[d];
      }
      fes.GroupComm().Reduce<int>(multiplicity, GroupCommunicator::Sum);
      fes.GroupComm().Bcast(multiplicity);

      // Compute offsets for the broken space.
      HYPRE_BigInt size = restriction.Height();
      Array<HYPRE_BigInt> flux_offsets;
      Array<HYPRE_BigInt> *partitions[] = { &flux_offsets };
      fes.GetParMesh()->GenerateOffsets(1, &size, partitions);
      num_global_flux_dofs = flux_offsets[2];

      // Same-order RT and DG interface spaces share local face DOF numbering:
      // the RT element-interior DOFs follow this common prefix. The interface
      // restriction selects interior faces, including shared faces, without
      // applying normal-orientation signs to the scalar multipliers.
      const auto &face_restriction =
         dynamic_cast<const L2InterfaceFaceRestriction&>(
            *trace_fes.GetFaceRestriction(ElementDofOrdering::NATIVE,
                                          FaceType::Interior));
      const Array<int> &interior = face_restriction.GatherMap();
      const int trace_size = trace_fes.GetVSize();

      // Keep all interface true DOFs. Inactive boundary multipliers are
      // eliminated from the system by replacing the equations with identity.
      num_global_trace_dofs = trace_prolongation.GetGlobalNumCols();
      trace_global = -1;
      for (int d = 0; d < trace_size; ++d)
      {
         const int owned = trace_fes.GetLocalTDofNumber(d);
         trace_local[d] = owned;
         if (interior[d] >= 0)
         {
            trace_global[d] = trace_fes.GetGlobalTDofNumber(d);
         }
         else if (owned >= 0) { inactive_trace.Append(owned); }
      }

      SparseMatrix local_Ct(dofs.Size(), trace_fes.GetVSize());
      for (int i = 0; i < dofs.Size(); ++i)
      {
         const int td = UnsignIndex(dofs[i]);
         if (td < trace_size && interior[td] >= 0)
         {
            local_Ct.Add(i, td, 1.0);
         }
      }
      local_Ct.Finalize();
      HypreParMatrix parallel_Ct(comm, num_global_flux_dofs,
                                 trace_fes.GlobalVSize(), flux_offsets.GetData(),
                                 trace_fes.GetDofOffsets(), &local_Ct);
      Ct.reset(ParMult(&parallel_Ct, &trace_prolongation, true));
      C.reset(new TransposeOperator(Ct.get()));
   }

   void Prolongate(const Vector &x, Vector &broken) const
   {
      Vector conforming(fes.GetVSize());
      prolongation.Mult(x, conforming);
      broken.SetSize(restriction.Height());
      restriction.Mult(conforming, broken);
   }

   void Assemble(const Vector &broken, Vector &x) const
   {
      Vector conforming(fes.GetVSize());
      restriction.MultTranspose(broken, conforming);
      x.SetSize(fes.GetTrueVSize());
      prolongation.MultTranspose(conforming, x);
   }

   void Average(const Vector &broken, Vector &x) const
   {
      Assemble(broken, x);
      for (int d = 0; d < multiplicity.Size(); ++d)
      {
         const int tdof = fes.GetLocalTDofNumber(d);
         if (tdof >= 0 && multiplicity[d]) { x(tdof) /= multiplicity[d]; }
      }
   }

   // R^T = P D^{-1}, the transpose of Average.
   void AverageTranspose(const Vector &x, Vector &broken) const
   {
      Vector scaled(x);
      for (int d = 0; d < multiplicity.Size(); ++d)
      {
         const int tdof = fes.GetLocalTDofNumber(d);
         if (tdof >= 0 && multiplicity[d]) { scaled(tdof) /= multiplicity[d]; }
      }
      Prolongate(scaled, broken);
   }
};

} // namespace mfem

#endif // MFEM_HDIV_HB_HPP
