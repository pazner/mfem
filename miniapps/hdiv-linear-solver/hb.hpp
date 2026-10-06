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

#include "discrete_divergence.hpp"
#include <vector>

namespace mfem
{

inline real_t Sign(int d) { return d < 0 ? -1.0 : 1.0; }

class BrokenSpace
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
   std::unique_ptr<HypreParMatrix> C;

public:
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
      C.reset(Ct->Transpose());
   }

   int GetNumBrokenDofs() const
   {
      return restriction.Height();
   }

   ParFiniteElementSpace &GetFESpace() const { return fes; }

   int GetNumTraceDofs() const
   {
      return trace_fes.GetTrueVSize();
   }

   int GetNumDofsPerElem() const
   {
      return nd;
   }

   HypreParMatrix &GetC()
   {
      return *C;
   }

   HypreParMatrix &GetCt()
   {
      return *Ct;
   }

   const Array<int> &GetInactiveTraceDofs() const
   {
      return inactive_trace;
   }

   void LocalProlongate(const Vector &x, Vector &broken) const
   {
      broken.SetSize(restriction.Height());
      restriction.Mult(x, broken);
   }

   void Prolongate(const Vector &x, Vector &broken) const
   {
      Vector conforming(fes.GetVSize());
      prolongation.Mult(x, conforming);
      LocalProlongate(conforming, broken);
   }

   /// Map conforming essential true DOFs to native broken DOF indices.
   void GetBrokenEssentialDofs(const Array<int> &ess_tdofs,
                               Array<int> &ess_broken_dofs) const
   {
      Vector essential(fes.GetTrueVSize()), broken_essential;
      essential = 0.0;
      for (int td : ess_tdofs) { essential(td) = 1.0; }
      Prolongate(essential, broken_essential);
      ess_broken_dofs.SetSize(0);
      for (int i = 0; i < broken_essential.Size(); ++i)
      {
         if (broken_essential(i) != 0.0) { ess_broken_dofs.Append(i); }
      }
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

/// Assemble W^{-1} + D M_g^{-1} D^T with unit mass coefficients.
/// Essential RT columns are eliminated; the caller owns the returned matrix.
inline HypreParMatrix *AssembleGraphLaplacian(
   ParFiniteElementSpace &fes_rt, ParFiniteElementSpace &fes_l2,
   const Array<int> &ess_rt_dofs)
{
   ParBilinearForm mass_rt(&fes_rt);
   mass_rt.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass_rt.AddDomainIntegrator(new VectorFEMassIntegrator);
   mass_rt.Assemble();
   Vector rt_mass_diag_inv(fes_rt.GetTrueVSize());
   mass_rt.AssembleDiagonal(rt_mass_diag_inv);
   rt_mass_diag_inv.Reciprocal();

   ParBilinearForm mass_l2(&fes_l2);
   mass_l2.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass_l2.AddDomainIntegrator(new MassIntegrator);
   mass_l2.Assemble();
   Vector l2_mass_diag_inv(fes_l2.GetTrueVSize());
   mass_l2.AssembleDiagonal(l2_mass_diag_inv);
   l2_mass_diag_inv.Reciprocal();

   SparseMatrix rt_diag(rt_mass_diag_inv), l2_diag(l2_mass_diag_inv);
   HypreParMatrix M_inverse(fes_rt.GetComm(), fes_rt.GlobalTrueVSize(),
                            fes_rt.GetTrueDofOffsets(), &rt_diag);
   HypreParMatrix W_inverse(fes_l2.GetComm(), fes_l2.GlobalTrueVSize(),
                            fes_l2.GetTrueDofOffsets(), &l2_diag);
   std::unique_ptr<HypreParMatrix> D(
      FormDiscreteDivergenceMatrix(fes_rt, fes_l2, ess_rt_dofs));
   std::unique_ptr<HypreParMatrix> Dt(D->Transpose());
   std::unique_ptr<HypreParMatrix> laplacian(RAP(&M_inverse, Dt.get()));
   HypreParMatrix *S = ParAdd(laplacian.get(), &W_inverse);
   S->CopyRowStarts();
   S->CopyColStarts();
   return S;
}

class GraphPreconditioner : public Solver
{
   Vector F_inverse;
   std::unique_ptr<HypreParMatrix> F_inverse_Et;
   std::unique_ptr<HypreParMatrix> S;
   std::unique_ptr<SparseMatrix> S_serial;
   std::unique_ptr<Solver> S_inv;
   mutable Vector z1, z2;

public:

   enum SolverType
   {
      AMG,
      DIRECT
   };

   GraphPreconditioner(BrokenSpace &fes_hat, SolverType stype=AMG)
      : Solver(fes_hat.GetNumTraceDofs()), F_inverse(height)
   {
      ParFiniteElementSpace &fes_rt = fes_hat.GetFESpace();

      // Compute the diagonal of the RT mass matrix using PA
      Vector diagonal_lex(fes_hat.GetNumBrokenDofs());
      diagonal_lex = 0.0;
      VectorFEMassIntegrator mass;
      mass.AssemblePA(fes_rt);
      mass.AssembleDiagonalPA(diagonal_lex);

      // Convert from lexicographic to native ordering
      const Array<int> &dof_map = dynamic_cast<const TensorBasisElement&>(
                                     *fes_rt.GetTypicalFE()).GetDofMap();
      const int ndof_per_el = fes_hat.GetNumDofsPerElem();
      Vector inverse_mass(fes_hat.GetNumBrokenDofs());
      for (int e = 0; e < fes_rt.GetNE(); ++e)
      {
         for (int i = 0; i < ndof_per_el; ++i)
         {
            const real_t m = diagonal_lex(e*ndof_per_el+i);
            inverse_mass(e*ndof_per_el+UnsignIndex(dof_map[i])) = 1.0/m;
         }
      }

      // Each broken flux DOF couples to at most one multiplier, with value 1.
      fes_hat.GetC().Mult(inverse_mass, F_inverse);
      // Inactive multipliers have identity equations in the constrained system.
      for (int td : fes_hat.GetInactiveTraceDofs()) { F_inverse(td) = 1.0; }
      F_inverse.Reciprocal();

      const int dim = fes_rt.GetMesh()->Dimension();
      const int order = fes_rt.GetMaxElementOrder();
      Array<int> element2face;
      FormElementToFace(dim, order, element2face);
      const int nvol_per_el = element2face.Size()/(2*dim);
      L2_FECollection scalar_fec(order-1, dim, BasisType::IntegratedGLL,
                                 FiniteElement::INTEGRAL);
      ParFiniteElementSpace scalar_fes(fes_rt.GetParMesh(), &scalar_fec);

      // Form M_b^{-1} D_hat^T in native broken coordinates.
      SparseMatrix Dt_mass(fes_hat.GetNumBrokenDofs(), scalar_fes.GetVSize());
      for (int e = 0; e < fes_rt.GetNE(); ++e)
      {
         for (int i = 0; i < nvol_per_el; ++i)
         {
            for (int k = 0; k < 2*dim; ++k)
            {
               const int d = element2face[k+2*dim*i];
               const int mapped = dof_map[UnsignIndex(d)];
               const int row = e*ndof_per_el+UnsignIndex(mapped);
               Dt_mass.Add(row, e*nvol_per_el+i,
                           Sign(d)*Sign(mapped)*inverse_mass(row));
            }
         }
      }
      Dt_mass.Finalize();
      HypreParMatrix &Ct = fes_hat.GetCt();
      HypreParMatrix Dt_par(fes_rt.GetComm(), Ct.GetGlobalNumRows(),
                            scalar_fes.GlobalVSize(), Ct.GetRowStarts(),
                            scalar_fes.GetDofOffsets(), &Dt_mass);
      F_inverse_Et.reset(ParMult(&fes_hat.GetC(), &Dt_par, true));
      F_inverse_Et->ScaleRows(F_inverse);

      Array<int> ess_rt_dofs;
      fes_rt.GetBoundaryTrueDofs(ess_rt_dofs);
      S.reset(AssembleGraphLaplacian(fes_rt, scalar_fes, ess_rt_dofs));

      if (stype == AMG)
      {
         S_inv.reset(new HypreBoomerAMG(*S));
      }
      else
      {
         S_serial = std::make_unique<SparseMatrix>();
         S->GetDiag(*S_serial);
         S_inv.reset(new UMFPackSolver(*S_serial));
      }
      z1.SetSize(S->Height());
      z2.SetSize(S->Height());
   }

   void SetOperator(const Operator &) override { }

   // F^{-1} + F^{-1} E^T S^{-1} E F^{-1}.
   void Mult(const Vector &x, Vector &y) const override
   {
      F_inverse_Et->MultTranspose(x, z1);
      z2 = 0.0;
      S_inv->Mult(z1, z2);
      y = x;
      y *= F_inverse;
      F_inverse_Et->AddMult(z2, y);
   }
};

/// Apply element inverses in native, element-contiguous broken coordinates.
/// Essential DOFs use the same broken numbering and are constrained before LU.
class BrokenElementSolver : public Solver
{
   DenseTensor lu;
   Array<int> pivots;
   std::vector<LUFactors> factors;

public:
   BrokenElementSolver(BilinearForm &a,
                       const Array<int> &ess_dofs = Array<int>())
      : Solver(0), lu(a.GetElementMatrices()),
        pivots(lu.SizeI()*lu.SizeK()), factors(lu.SizeK())
   {
      const int nd = lu.SizeI();
      height = width = nd*lu.SizeK();
      for (int d : ess_dofs)
      {
         const int e = d/nd, j = d%nd;
         for (int i = 0; i < nd; ++i) { lu(i,j,e) = lu(j,i,e) = 0.0; }
         lu(j,j,e) = 1.0;
      }
      for (int e = 0; e < lu.SizeK(); ++e)
      {
         factors[e] = LUFactors(lu.GetData(e), pivots.GetData()+e*nd);
         MFEM_VERIFY(factors[e].Factor(nd),
                     "Singular element matrix on element " << e);
      }
   }

   BrokenElementSolver(const BrokenElementSolver &) = delete;
   BrokenElementSolver &operator=(const BrokenElementSolver &) = delete;

   void SetOperator(const Operator &) override { }

   void Mult(const Vector &x, Vector &y) const override
   {
      y = x;
      real_t *data = y.HostReadWrite();
      const int nd = lu.SizeI();
      for (int e = 0; e < lu.SizeK(); ++e)
      {
         factors[e].Solve(nd, 1, data+e*nd);
      }
   }
};

} // namespace mfem

#endif // MFEM_HDIV_HB_HPP
