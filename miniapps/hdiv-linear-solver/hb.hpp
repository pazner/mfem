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
#include "fast_diag.hpp"
#include <vector>

namespace mfem
{

inline real_t Sign(int d) { return d < 0 ? -1.0 : 1.0; }

enum SolverType { AMG, DIRECT };

/// Set y(idx[i]) = x(idx[i]) for all i, on the device.
inline void CopyEntries(const Array<int> &idx, const Vector &x, Vector &y)
{
   const int n = idx.Size();
   if (n == 0) { return; }
   const int *d_idx = idx.Read();
   const real_t *d_x = x.Read();
   real_t *d_y = y.ReadWrite();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { d_y[d_idx[i]] = d_x[d_idx[i]]; });
}

/// Return the diagonal matrix with entries d, filled on the device.
inline std::unique_ptr<SparseMatrix> DiagonalSparseMatrix(const Vector &d)
{
   const int n = d.Size();
   auto D = std::make_unique<SparseMatrix>(n, n, 1);
   const real_t *d_d = d.Read();
   int *J = D->WriteJ();
   real_t *A = D->WriteData();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { J[i] = i; A[i] = d_d[i]; });
   return D;
}

/// Scale row i of the finalized matrix A by s(i), on the device.
inline void ScaleRowsDevice(SparseMatrix &A, const Vector &s)
{
   const int *I = A.ReadI();
   const real_t *d_s = s.Read();
   real_t *data = A.ReadWriteData();
   mfem::forall(A.Height(), [=] MFEM_HOST_DEVICE (int row)
   {
      for (int k = I[row]; k < I[row+1]; ++k) { data[k] *= d_s[row]; }
   });
}

/// Invert the lexicographic element diagonal d_lex into native broken order.
inline void InvertLexDiagonal(const Array<int> &dof_map, const Vector &d_lex,
                              Vector &inverse)
{
   const int nd = dof_map.Size();
   const int *map = dof_map.Read();
   const real_t *d = d_lex.Read();
   real_t *inv = inverse.Write();
   mfem::forall(d_lex.Size(), [=] MFEM_HOST_DEVICE (int j)
   {
      const int e = j/nd, i = j%nd;
      inv[e*nd+UnsignIndex(map[i])] = 1.0/d[j];
   });
}

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
   // Inverse multiplicity of each true DOF (one where it is not shared).
   Vector inverse_multiplicity;
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
      // Filled on the host from the host-side DOF maps, then used on device.
      inverse_multiplicity.SetSize(fes.GetTrueVSize());
      inverse_multiplicity = 1.0;
      for (int d = 0; d < multiplicity.Size(); ++d)
      {
         const int tdof = fes.GetLocalTDofNumber(d);
         if (tdof >= 0 && multiplicity[d])
         { inverse_multiplicity(tdof) = 1.0/multiplicity[d]; }
      }
      inverse_multiplicity.UseDevice(true);

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

      // The gather map may already have been used on the device.
      const int *h_dofs = dofs.HostRead();
      SparseMatrix local_Ct(dofs.Size(), trace_fes.GetVSize());
      for (int i = 0; i < dofs.Size(); ++i)
      {
         const int td = UnsignIndex(h_dofs[i]);
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
      conforming.UseDevice(true);
      prolongation.Mult(x, conforming);
      LocalProlongate(conforming, broken);
   }

   /// Map conforming essential true DOFs to native broken DOF indices.
   void GetBrokenEssentialDofs(const Array<int> &ess_tdofs,
                               Array<int> &ess_broken_dofs) const
   {
      Vector essential(fes.GetTrueVSize()), broken_essential;
      essential.UseDevice(true);
      broken_essential.UseDevice(true);
      essential = 0.0;
      essential.SetSubVector(ess_tdofs, 1.0);
      Prolongate(essential, broken_essential);
      // The index list is built on the host, where its consumers use it.
      const real_t *marked = broken_essential.HostRead();
      ess_broken_dofs.SetSize(0);
      for (int i = 0; i < broken_essential.Size(); ++i)
      {
         if (marked[i] != 0.0) { ess_broken_dofs.Append(i); }
      }
   }

   void Assemble(const Vector &broken, Vector &x) const
   {
      Vector conforming(fes.GetVSize());
      conforming.UseDevice(true);
      restriction.MultTranspose(broken, conforming);
      x.SetSize(fes.GetTrueVSize());
      prolongation.MultTranspose(conforming, x);
   }

   void Average(const Vector &broken, Vector &x) const
   {
      Assemble(broken, x);
      x *= inverse_multiplicity;
   }

   // R^T = P D^{-1}, the transpose of Average.
   void AverageTranspose(const Vector &x, Vector &broken) const
   {
      Vector scaled(x);
      scaled.UseDevice(true);
      scaled *= inverse_multiplicity;
      Prolongate(scaled, broken);
   }
};

/// Assemble W^{-1} + D M_g^{-1} D^T, with diagonal RT and L2 masses weighted
/// by alpha and beta (one when null). Essential RT columns are eliminated; the
/// caller owns the returned matrix.
inline HypreParMatrix *AssembleGraphLaplacian(
   ParFiniteElementSpace &fes_rt, ParFiniteElementSpace &fes_l2,
   const Array<int> &ess_rt_dofs, Coefficient *alpha = nullptr,
   Coefficient *beta = nullptr)
{
   ParBilinearForm mass_rt(&fes_rt);
   mass_rt.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   auto *rt_integrator = new VectorFEMassIntegrator(alpha);
   rt_integrator->SetIntRule(GetCoefficientRule(alpha));
   mass_rt.AddDomainIntegrator(rt_integrator);
   mass_rt.Assemble();
   Vector rt_mass_diag_inv(fes_rt.GetTrueVSize());
   rt_mass_diag_inv.UseDevice(true);
   mass_rt.AssembleDiagonal(rt_mass_diag_inv);
   rt_mass_diag_inv.Reciprocal();

   ParBilinearForm mass_l2(&fes_l2);
   mass_l2.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass_l2.AddDomainIntegrator(beta ? new MassIntegrator(*beta,
                                                         GetCoefficientRule(beta)) :
                               new MassIntegrator);
   mass_l2.Assemble();
   Vector l2_mass_diag_inv(fes_l2.GetTrueVSize());
   l2_mass_diag_inv.UseDevice(true);
   mass_l2.AssembleDiagonal(l2_mass_diag_inv);
   l2_mass_diag_inv.Reciprocal();

   auto rt_diag = DiagonalSparseMatrix(rt_mass_diag_inv);
   auto l2_diag = DiagonalSparseMatrix(l2_mass_diag_inv);
   HypreParMatrix M_inverse(fes_rt.GetComm(), fes_rt.GlobalTrueVSize(),
                            fes_rt.GetTrueDofOffsets(), rt_diag.get());
   HypreParMatrix W_inverse(fes_l2.GetComm(), fes_l2.GlobalTrueVSize(),
                            fes_l2.GetTrueDofOffsets(), l2_diag.get());
   std::unique_ptr<HypreParMatrix> D(
      FormDiscreteDivergenceMatrix(fes_rt, fes_l2, ess_rt_dofs));
   std::unique_ptr<HypreParMatrix> Dt(D->Transpose());
   std::unique_ptr<HypreParMatrix> laplacian(RAP(&M_inverse, Dt.get()));
   HypreParMatrix *S = ParAdd(laplacian.get(), &W_inverse);
   S->CopyRowStarts();
   S->CopyColStarts();
   return S;
}

/// Preconditioner for the multiplier Schur complement C A^{-1} C^T, where
/// A = alpha (u,v) + beta (div u,div v) on the broken RT space. Uses F^{-1} +
/// F^{-1} E^T S^{-1} E F^{-1}, with lumped alpha-weighted RT and beta-weighted
/// L2 masses. The coefficients default to one when null; beta must be positive.
class GraphPreconditioner : public Solver
{
   Vector F_inverse;
   std::unique_ptr<HypreParMatrix> F_inverse_Et;
   std::unique_ptr<HypreParMatrix> S;
   std::unique_ptr<SparseMatrix> S_serial;
   std::unique_ptr<Solver> S_inv;
   mutable Vector z1, z2;
public:
   GraphPreconditioner(BrokenSpace &fes_hat, const Array<int> &ess_rt_dofs,
                       Coefficient *alpha = nullptr, Coefficient *beta = nullptr,
                       SolverType stype=AMG)
      : Solver(fes_hat.GetNumTraceDofs()), F_inverse(height)
   {
      ParFiniteElementSpace &fes_rt = fes_hat.GetFESpace();
      F_inverse.UseDevice(true);

      // Compute the diagonal of the alpha-weighted RT mass matrix using PA
      Vector diagonal_lex(fes_hat.GetNumBrokenDofs());
      diagonal_lex.UseDevice(true);
      diagonal_lex = 0.0;
      VectorFEMassIntegrator mass(alpha);
      mass.SetIntRule(GetCoefficientRule(alpha));
      mass.AssemblePA(fes_rt);
      mass.AssembleDiagonalPA(diagonal_lex);

      // Convert from lexicographic to native ordering
      const Array<int> &dof_map = dynamic_cast<const TensorBasisElement&>(
                                     *fes_rt.GetTypicalFE()).GetDofMap();
      const int ndof_per_el = fes_hat.GetNumDofsPerElem();
      const int n_broken = fes_hat.GetNumBrokenDofs();
      Vector inverse_mass(n_broken);
      inverse_mass.UseDevice(true);
      InvertLexDiagonal(dof_map, diagonal_lex, inverse_mass);

      // Each broken flux DOF couples to at most one multiplier, with value 1.
      fes_hat.GetC().Mult(inverse_mass, F_inverse);
      // Inactive multipliers have identity equations in the constrained system.
      F_inverse.SetSubVector(fes_hat.GetInactiveTraceDofs(), 1.0);
      F_inverse.Reciprocal();

      const int dim = fes_rt.GetMesh()->Dimension();
      const int order = fes_rt.GetMaxElementOrder();
      Array<int> element2face;
      FormElementToFace(dim, order, element2face);
      const int nvol_per_el = element2face.Size()/(2*dim);
      L2_FECollection scalar_fec(order-1, dim, BasisType::IntegratedGLL,
                                 FiniteElement::INTEGRAL);
      ParFiniteElementSpace scalar_fes(fes_rt.GetParMesh(), &scalar_fec);

      // Form M_b^{-1} D_hat^T in native broken coordinates. The signed pattern
      // is built on the host; the rows are scaled by M_b^{-1} on the device.
      SparseMatrix Dt_mass(n_broken, scalar_fes.GetVSize());
      const int *h_map = dof_map.HostRead();
      for (int e = 0; e < fes_rt.GetNE(); ++e)
      {
         for (int i = 0; i < nvol_per_el; ++i)
         {
            for (int k = 0; k < 2*dim; ++k)
            {
               const int d = element2face[k+2*dim*i];
               const int mapped = h_map[UnsignIndex(d)];
               const int row = e*ndof_per_el+UnsignIndex(mapped);
               Dt_mass.Add(row, e*nvol_per_el+i, Sign(d)*Sign(mapped));
            }
         }
      }
      Dt_mass.Finalize();
      ScaleRowsDevice(Dt_mass, inverse_mass);
      HypreParMatrix &Ct = fes_hat.GetCt();
      HypreParMatrix Dt_par(fes_rt.GetComm(), Ct.GetGlobalNumRows(),
                            scalar_fes.GlobalVSize(), Ct.GetRowStarts(),
                            scalar_fes.GetDofOffsets(), &Dt_mass);
      F_inverse_Et.reset(ParMult(&fes_hat.GetC(), &Dt_par, true));
      F_inverse_Et->ScaleRows(F_inverse);

      S.reset(AssembleGraphLaplacian(fes_rt, scalar_fes, ess_rt_dofs, alpha,
                                     beta));

      if (stype == AMG)
      {
         S_inv.reset(new HypreBoomerAMG(*S));
      }
      else
      {
#ifdef MFEM_USE_SUITESPARSE
         S_serial = std::make_unique<SparseMatrix>();
         S->GetDiag(*S_serial);
         S_inv.reset(new UMFPackSolver(*S_serial));
#else
         MFEM_ABORT("The direct solver requires MFEM_USE_SUITESPARSE.");
#endif
      }
      z1.SetSize(S->Height());
      z2.SetSize(S->Height());
      z1.UseDevice(true);
      z2.UseDevice(true);
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

/// Apply unmarked domain integrators using PA in native broken RT coordinates.
class BrokenElementPAOperator : public Operator
{
   const Array<BilinearFormIntegrator*> &integrators;
   const Array<int> &dof_map;
   mutable Vector x_lex, y_lex;

public:
   BrokenElementPAOperator(BilinearForm &a)
      : Operator(a.FESpace()->GetNE()*a.FESpace()->GetTypicalFE()->GetDof()),
        integrators(*a.GetDBFI()),
        dof_map(dynamic_cast<const TensorBasisElement&>(
                   *a.FESpace()->GetTypicalFE()).GetDofMap()),
        x_lex(height), y_lex(height)
   {
      x_lex.UseDevice(true);
      y_lex.UseDevice(true);
      for (auto *integrator : integrators) { integrator->AssemblePA(*a.FESpace()); }
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      const int nd = dof_map.Size();
      const int *map = dof_map.Read();
      const real_t *input = x.Read();
      real_t *lex_input = x_lex.Write();
      MFEM_FORALL(i, height,
      {
         const int d = map[i%nd];
         lex_input[i] = (d < 0 ? -1.0 : 1.0)*input[(i/nd)*nd+UnsignIndex(d)];
      });
      y_lex = 0.0;
      for (auto *integrator : integrators) { integrator->AddMultPA(x_lex, y_lex); }
      y.SetSize(height);
      const real_t *lex_output = y_lex.Read();
      real_t *output = y.Write();
      MFEM_FORALL(i, height,
      {
         const int d = map[i%nd];
         output[(i/nd)*nd+UnsignIndex(d)] = (d < 0 ? -1.0 : 1.0)*lex_output[i];
      });
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
      // The essential DOF list may already have been used on the device.
      const int *h_ess = ess_dofs.HostRead();
      for (int k = 0; k < ess_dofs.Size(); ++k)
      {
         const int d = h_ess[k], e = d/nd, j = d%nd;
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

   void SolveElement(int e, DenseMatrix &rhs) const
   {
      factors[e].Solve(lu.SizeI(), rhs.Width(), rhs.Data());
   }

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

/// Solve the weak multiplier system assembled by MFEM hybridization.
/** In LOR mode, the integrators of the high-order form are evaluated on the
    refined mesh by default, which requires coefficients that do not depend on
    the high-order quadrature (e.g. not QuadratureFunctionCoefficient). Passing
    lor_alpha and lor_beta instead assembles alpha (u,v) + beta (div u,div v)
    on the refined mesh with those coefficients. Element attributes are
    inherited by the refined mesh, so PWConstCoefficient may be used. */
class TraceMultiplierSolver : public Solver
{
   std::unique_ptr<HypreParMatrix> Ct;
   std::unique_ptr<HypreParMatrix> H;
   std::unique_ptr<HypreParMatrix> J;
   std::unique_ptr<SparseMatrix> H_serial;
   std::unique_ptr<Solver> H_inv;
   mutable Vector lor_rhs, lor_solution;

   class LORSpace : public ParLORDiscretization
   {
   public:
      using ParLORDiscretization::ParLORDiscretization;
      void GetLocalPermutation(Array<int> &perm) const
      { ConstructLocalDofPermutation(perm); }
   };

   // Inject dual weak coordinates on macro faces; subcell-interior rows are zero.
   static HypreParMatrix *BuildMultiplierInjection(
      ParFiniteElementSpace &fes, ParFiniteElementSpace &trace_fes,
      const SparseMatrix &local_Ct, ParFiniteElementSpace &lor_fes,
      ParFiniteElementSpace &lor_trace_fes, const SparseMatrix &lor_Ct,
      const Array<int> &perm)
   {
      Array<int> lor_dof(fes.GetVSize());
      lor_dof = -1;
      for (int d = 0; d < perm.Size(); ++d)
      { lor_dof[UnsignIndex(perm[d])] = d; }
      const auto &restriction = dynamic_cast<const ElementRestriction&>(
                                   *lor_fes.GetElementRestriction(
                                      ElementDofOrdering::NATIVE));
      const Array<int> &dofs = restriction.GatherMap();
      Array<int> first_copy(lor_fes.GetVSize());
      first_copy = -1;
      for (int i = 0; i < dofs.Size(); ++i)
      {
         const int d = UnsignIndex(dofs[i]);
         if (first_copy[d] < 0) { first_copy[d] = i; }
      }
      const auto &cf = lor_fes.GetMesh()->GetRefinementTransforms();
      const int nd = fes.GetTypicalFE()->GetDof();
      const int lor_nd = lor_fes.GetTypicalFE()->GetDof();
      SparseMatrix local_J(lor_trace_fes.GetTrueVSize(), trace_fes.GetVSize());
      Array<int> flux_dofs, trace_dofs, element_dofs, rows;
      DenseMatrix face_Ct, inverse;
      for (int f = 0; f < fes.GetMesh()->GetNumFaces(); ++f)
      {
         int e, e2;
         fes.GetMesh()->GetFaceElements(f, &e, &e2);
         fes.GetFaceVDofs(f, flux_dofs);
         trace_fes.GetFaceVDofs(f, trace_dofs);
         fes.GetElementVDofs(e, element_dofs);
         rows.SetSize(flux_dofs.Size());
         for (int i = 0; i < rows.Size(); ++i)
         {
            int j = 0;
            while (UnsignIndex(element_dofs[j]) != UnsignIndex(flux_dofs[i]))
            { ++j; }
            rows[i] = e*nd+j;
         }
         face_Ct.SetSize(rows.Size(), trace_dofs.Size());
         local_Ct.GetSubMatrix(rows, trace_dofs, face_Ct);
         // Express the weak face constraint in globally oriented flux DOFs.
         for (int i = 0; i < rows.Size(); ++i)
         {
            for (int j = 0; j < trace_dofs.Size(); ++j)
            { face_Ct(i,j) *= Sign(element_dofs[rows[i]-e*nd]); }
         }
         const bool active = face_Ct.MaxMaxNorm() != 0.0;
         if (active) { DenseMatrixInverse(face_Ct).GetInverseMatrix(inverse); }
         // Preserve identity equations for inactive boundary multipliers.
         else { inverse.Diag(1.0, rows.Size()); }
         for (int i = 0; i < rows.Size(); ++i)
         {
            const int ld = lor_dof[UnsignIndex(flux_dofs[i])];
            const int row = lor_trace_fes.GetLocalTDofNumber(ld);
            if (row < 0) { continue; }
            real_t scale = 1.0;
            if (active)
            {
               const int copy = first_copy[ld];
               scale = Sign(dofs[copy])*Sign(perm[ld])*lor_Ct(copy, ld);
               if (cf.embeddings[copy/lor_nd].parent != e) { scale = -scale; }
            }
            for (int j = 0; j < trace_dofs.Size(); ++j)
            { local_J.Add(row, trace_dofs[j], scale*inverse(j,i)); }
         }
      }
      local_J.Finalize();
      Array<HYPRE_BigInt> columns(local_J.NumNonZeroElems());
      for (int i = 0; i < columns.Size(); ++i)
      { columns[i] = trace_fes.GetGlobalTDofNumber(local_J.GetJ()[i]); }
      return new HypreParMatrix(fes.GetComm(), local_J.Height(),
                                lor_trace_fes.GlobalTrueVSize(),
                                trace_fes.GlobalTrueVSize(), local_J.GetI(),
                                columns.GetData(), local_J.GetData(),
                                lor_trace_fes.GetTrueDofOffsets(),
                                trace_fes.GetTrueDofOffsets());
   }

public:
   enum AssemblyMode { ACTUAL, LOR };

   TraceMultiplierSolver(ParBilinearForm &a, const Array<int> &ess_tdofs,
                         AssemblyMode amode=ACTUAL, SolverType stype=AMG,
                         Coefficient *lor_alpha = nullptr,
                         Coefficient *lor_beta = nullptr)
      : Solver(0)
   {
      MFEM_VERIFY((lor_alpha == nullptr) == (lor_beta == nullptr),
                  "Specify both or neither LOR coefficients.");
      MFEM_VERIFY(!lor_alpha || amode == LOR,
                  "LOR coefficients require LOR assembly mode.");
      ParFiniteElementSpace &fes = *a.ParFESpace();
      DG_Interface_FECollection trace_fec(fes.GetMaxElementOrder()-1,
                                          fes.GetMesh()->Dimension());
      ParFiniteElementSpace trace_fes(fes.GetParMesh(), &trace_fec);
      Hybridization hybridization(&fes, &trace_fes);
      hybridization.SetConstraintIntegrator(new NormalTraceJumpIntegrator);
      ess_tdofs.HostRead(); // Init reads the list on the host.
      hybridization.Init(ess_tdofs);
      SparseMatrix &local_Ct = hybridization.GetConstraintTranspose();
      HYPRE_BigInt size = local_Ct.Height();
      Array<HYPRE_BigInt> flux_offsets;
      Array<HYPRE_BigInt> *partitions[] = { &flux_offsets };
      fes.GetParMesh()->GenerateOffsets(1, &size, partitions);
      HypreParMatrix parallel_Ct(fes.GetComm(), flux_offsets.Last(),
                                 trace_fes.GlobalVSize(), flux_offsets.GetData(),
                                 trace_fes.GetDofOffsets(), &local_Ct);
      Ct.reset(ParMult(&parallel_Ct, trace_fes.Dof_TrueDof_Matrix(), true));
      height = width = trace_fes.GetTrueVSize();
      if (amode == ACTUAL)
      {
         hybridization.AssembleElementMatrices(a.GetElementMatrices());
         hybridization.Finalize();
         H.reset(new HypreParMatrix(hybridization.GetParallelMatrix()));
      }
      else
      {
         MFEM_VERIFY(!fes.GetParMesh()->Nonconforming(),
                     "LOR multiplier transfer requires a conforming mesh.");
         LORSpace lor(fes);
         Array<int> perm;
         lor.GetLocalPermutation(perm);
         // Use native RT0 ownership, rather than the copied HO prolongation.
         ParFiniteElementSpace lor_fes(lor.GetParFESpace().GetParMesh(),
                                       lor.GetParFESpace().FEColl());
         DG_Interface_FECollection lor_trace_fec(0, fes.GetMesh()->Dimension());
         ParFiniteElementSpace lor_trace_fes(lor_fes.GetParMesh(), &lor_trace_fec);
         Vector essential(fes.GetTrueVSize()), local_essential(fes.GetVSize());
         essential.UseDevice(true);
         local_essential.UseDevice(true);
         essential = 0.0;
         essential.SetSubVector(ess_tdofs, 1.0);
         fes.GetProlongationMatrix()->Mult(essential, local_essential);
         // The index list is built on the host, where Hybridization uses it.
         const real_t *marked = local_essential.HostRead();
         Array<int> lor_ess;
         for (int d = 0; d < perm.Size(); ++d)
         {
            const int td = lor_fes.GetLocalTDofNumber(d);
            if (td >= 0 && marked[UnsignIndex(perm[d])] != 0.0)
            { lor_ess.Append(td); }
         }
         Hybridization lor_hybridization(&lor_fes, &lor_trace_fes);
         lor_hybridization.SetConstraintIntegrator(new NormalTraceJumpIntegrator);
         lor_hybridization.Init(lor_ess);
         J.reset(BuildMultiplierInjection(fes, trace_fes, local_Ct, lor_fes,
                                          lor_trace_fes,
                                          lor_hybridization.GetConstraintTranspose(),
                                          perm));
         ParBilinearForm lor_a(&lor_fes);
         IntegrationRules irs(0, Quadrature1D::GaussLobatto);
         const IntegrationRule &rule = irs.Get(
                                          lor_fes.GetMesh()->GetTypicalElementGeometry(), 1);
         Array<const IntegrationRule*> saved_rules;
         if (lor_alpha)
         {
            lor_a.AddDomainIntegrator(new VectorFEMassIntegrator(*lor_alpha));
            lor_a.AddDomainIntegrator(new DivDivIntegrator(*lor_beta));
            for (auto *integrator : *lor_a.GetDBFI())
            { integrator->SetIntRule(&rule); }
         }
         else
         {
            lor_a.UseExternalIntegrators();
            for (auto *integrator : *a.GetDBFI())
            {
               saved_rules.Append(integrator->GetIntRule());
               integrator->SetIntRule(&rule);
               lor_a.AddDomainIntegrator(integrator);
            }
         }
         lor_hybridization.AssembleElementMatrices(lor_a.GetElementMatrices());
         for (int i = 0; i < saved_rules.Size(); ++i)
         { (*a.GetDBFI())[i]->SetIntRule(saved_rules[i]); }
         lor_hybridization.Finalize();
         H.reset(new HypreParMatrix(lor_hybridization.GetParallelMatrix()));
         lor_rhs.SetSize(H->Height());
         lor_solution.SetSize(H->Height());
      }

      if (stype == AMG)
      {
         auto *amg = new HypreBoomerAMG(*H);
         H_inv.reset(amg);
      }
      else
      {
         MFEM_VERIFY(Mpi::WorldSize() == 1,
                     "The direct multiplier solver requires one MPI rank.");
#ifdef MFEM_USE_SUITESPARSE
         H_serial.reset(new SparseMatrix);
         H->GetDiag(*H_serial);
         H_inv.reset(new UMFPackSolver(*H_serial));
#else
         MFEM_ABORT("The direct solver requires MFEM_USE_SUITESPARSE.");
#endif
      }
   }

   HypreParMatrix &GetConstraintTranspose() { return *Ct; }
   const HypreParMatrix &GetConstraintTranspose() const { return *Ct; }
   /// Return the full refined multiplier matrix in LOR mode.
   const HypreParMatrix &GetMatrix() const { return *H; }
   void SetOperator(const Operator &) override { }
   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(height);
      if (J)
      {
         J->Mult(x, lor_rhs);
         H_inv->Mult(lor_rhs, lor_solution);
         J->MultTranspose(lor_solution, y);
      }
      else { H_inv->Mult(x, y); }
   }
};

} // namespace mfem

#endif // MFEM_HDIV_HB_HPP
