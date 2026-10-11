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
#include "general/scan.hpp"
#include <vector>

namespace mfem
{

inline MFEM_HOST_DEVICE real_t Sign(int d) { return d < 0 ? -1.0 : 1.0; }

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

/// Set y(i) = a(i) x(i), on the device.
inline void MultiplyEntries(const Vector &a, const Vector &x, Vector &y)
{
   const int n = x.Size();
   const real_t *d_a = a.Read();
   const real_t *d_x = x.Read();
   real_t *d_y = y.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { d_y[i] = d_a[i]*d_x[i]; });
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

/// Lexicographic DOF map of the typical (tensor-product) element of fes.
inline const Array<int> &GetDofMap(const FiniteElementSpace &fes)
{
   return dynamic_cast<const TensorBasisElement&>(*fes.GetTypicalFE())
          .GetDofMap();
}

/// Convert a native element matrix to the lexicographic broken coordinates,
/// u_lex(j) = sign(dof_map[j]) u(|dof_map[j]|), as A_lex = P A P^T.
inline void NativeToLexicographic(const Array<int> &dof_map,
                                  const DenseMatrix &A, DenseMatrix &A_lex)
{
   const int nd = dof_map.Size();
   const int *map = dof_map.HostRead();
   A_lex.SetSize(nd);
   for (int k = 0; k < nd; ++k)
   {
      for (int j = 0; j < nd; ++j)
      {
         A_lex(j,k) = Sign(map[j])*Sign(map[k])*
                      A(UnsignIndex(map[j]), UnsignIndex(map[k]));
      }
   }
}

/// Convert the rows of a native broken matrix (rows e*nd + i) to the
/// lexicographic broken coordinates, returning P A.
inline std::unique_ptr<SparseMatrix> NativeToLexicographicRows(
   const Array<int> &dof_map, const SparseMatrix &A)
{
   const int nd = dof_map.Size();
   const int n = A.Height();
   const int *map = dof_map.HostRead();
   SparseMatrix P(n, n);
   for (int i = 0; i < n; ++i)
   {
      const int d = map[i%nd];
      P.Set(i, (i/nd)*nd + UnsignIndex(d), Sign(d));
   }
   P.Finalize();
   return std::unique_ptr<SparseMatrix>(Mult(P, A));
}

/// Set y(i) = sign(dof_map[i % nd]) x(i), on the device.
inline void MultiplyDofMapSigns(const Array<int> &dof_map, const Vector &x,
                                Vector &y)
{
   const int nd = dof_map.Size();
   const int *map = dof_map.Read();
   const real_t *d_x = x.Read();
   y.SetSize(x.Size());
   real_t *d_y = y.Write();
   mfem::forall(x.Size(), [=] MFEM_HOST_DEVICE (int i)
   { d_y[i] = Sign(map[i%nd])*d_x[i]; });
}

/// Set idx to the indices i with flags(i) != 0, compacted on the device.
inline void FlaggedIndices(const Vector &flags, Array<int> &idx)
{
   const int n = flags.Size();
   Array<int> iota(n), count(1);
   int *d_iota = iota.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { d_iota[i] = i; });
   idx.SetSize(n);
   CopyFlagged(true, iota.Read(), flags.Read(), idx.Write(), count.Write(), n);
   idx.SetSize(count.HostRead()[0]);
}

/// Replace each positive count c by 1/c, and nonpositive counts by one.
inline void InvertCounts(Vector &counts)
{
   real_t *d_c = counts.ReadWrite();
   mfem::forall(counts.Size(), [=] MFEM_HOST_DEVICE (int i)
   { d_c[i] = d_c[i] > 0.0 ? 1.0/d_c[i] : 1.0; });
}

/// For i < n, set flags(i) to one where map[i] < 0, and zero otherwise.
inline void FlagNegative(const Array<int> &map, int n, Vector &flags)
{
   flags.SetSize(n);
   const int *d_map = map.Read();
   real_t *d_flags = flags.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   { d_flags[i] = d_map[i] < 0 ? 1.0 : 0.0; });
}

/// Form the local constraint transpose on the device. Row i of the
/// lexicographic broken space couples to the trace DOF |dofs[i]| when that is a
/// face DOF (index < trace_size) on an interior face (interior >= 0), with
/// entry sign(dof_map[i % nd]) (unit in native element coordinates); rows of
/// other DOFs are empty. Row offsets are formed by a scan.
inline void FormLocalConstraintTranspose(const Array<int> &dofs,
                                         const Array<int> &dof_map,
                                         const Array<int> &interior,
                                         int trace_size, SparseMatrix &Ct)
{
   const int n = dofs.Size();
   const int nd = dof_map.Size();
   Ct.OverrideSize(n, trace_size);
   Ct.GetMemoryI().New(n + 1);
   Array<int> trace_col(n), nnz(1);
   {
      const int *d_dofs = dofs.Read();
      const int *d_interior = interior.Read();
      int *I = Ct.WriteI();
      int *col = trace_col.Write();
      mfem::forall(n + 1, [=] MFEM_HOST_DEVICE (int i)
      {
         if (i == n) { I[i] = 0; return; }
         const int td = UnsignIndex(d_dofs[i]);
         I[i] = (td < trace_size && d_interior[td] >= 0) ? 1 : 0;
         col[i] = td;
      });
      ExclusiveScan(true, I, I, n + 1, 0);
      int *d_nnz = nnz.Write();
      mfem::forall(1, [=] MFEM_HOST_DEVICE (int) { d_nnz[0] = I[n]; });
   }
   const int local_nnz = nnz.HostRead()[0];
   Ct.GetMemoryJ().New(local_nnz);
   Ct.GetMemoryData().New(local_nnz);
   const int *I = Ct.ReadI();
   const int *col = trace_col.Read();
   const int *map = dof_map.Read();
   int *J = Ct.WriteJ();
   real_t *A = Ct.WriteData();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      if (I[i+1] > I[i]) { J[I[i]] = col[i]; A[I[i]] = Sign(map[i%nd]); }
   });
}

/// Form D_hat M^{-1} on the device, where D_hat is the broken divergence
/// mapping lexicographic broken RT DOFs (nd per element) to the element-wise L2
/// subcells e*nvol+i, and M^{-1} is the diagonal inverse_mass. Each row has
/// exactly 2*dim entries.
inline void FormBrokenDivergence(const Array<int> &element2face, int nd,
                                 int dim, int ne, const Vector &inverse_mass,
                                 SparseMatrix &D)
{
   const int two_dim = 2*dim;
   const int nvol = element2face.Size()/two_dim;
   const int nrows = ne*nvol, nnz = nrows*two_dim;
   D.OverrideSize(nrows, ne*nd);
   D.GetMemoryI().New(nrows + 1);
   D.GetMemoryJ().New(nnz);
   D.GetMemoryData().New(nnz);
   int *I = D.WriteI();
   int *J = D.WriteJ();
   real_t *V = D.WriteData();
   const int *e2f = element2face.Read();
   const real_t *inv = inverse_mass.Read();
   mfem::forall(nrows + 1, [=] MFEM_HOST_DEVICE (int i) { I[i] = two_dim*i; });
   mfem::forall(nnz, [=] MFEM_HOST_DEVICE (int ii)
   {
      const int k = ii%two_dim, row = ii/two_dim;
      const int i = row%nvol, e = row/nvol;
      const int d = e2f[k+two_dim*i];
      const int col = e*nd+UnsignIndex(d);
      J[ii] = col;
      V[ii] = Sign(d)*inv[col];
   });
}

/// Broken RT space in lexicographic, element-contiguous coordinates: the
/// E-vector layout of MFEM's lexicographic element restriction, including the
/// orientation and DOF map signs, as consumed by partial assembly kernels.
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
   Array<int> inactive_trace;
   // Inverse multiplicity of each true DOF (one where it is not shared).
   Vector inverse_multiplicity;
   const HypreParMatrix &trace_prolongation;
   std::unique_ptr<HypreParMatrix> Ct;
   std::unique_ptr<HypreParMatrix> C;

public:
   BrokenSpace(ParFiniteElementSpace &fes_)
      : fes(fes_), trace_fec(fes.GetMaxElementOrder()-1, fes.GetMesh()->Dimension()),
        trace_fes(fes.GetParMesh(), &trace_fec),
        restriction(dynamic_cast<const ElementRestriction&>(
                       *fes.GetElementRestriction(
                          ElementDofOrdering::LEXICOGRAPHIC))),
        prolongation(*fes.GetProlongationMatrix()),
        nd(fes.GetTypicalFE()->GetDof()), dofs(restriction.GatherMap()),
        trace_prolongation(*trace_fes.Dof_TrueDof_Matrix())
   {
      const MPI_Comm comm = fes.GetComm();
      const int n_broken = restriction.Height();

      // Count the element copies of each true DOF over all ranks. The
      // restriction is applied without RT orientation signs; the prolongation
      // is boolean.
      {
         Vector ones(n_broken), local_count(fes.GetVSize());
         ones.UseDevice(true);
         local_count.UseDevice(true);
         ones = 1.0;
         restriction.AbsMultTranspose(ones, local_count);
         inverse_multiplicity.SetSize(fes.GetTrueVSize());
         inverse_multiplicity.UseDevice(true);
         prolongation.MultTranspose(local_count, inverse_multiplicity);
         InvertCounts(inverse_multiplicity);
      }

      // Compute offsets for the broken space.
      HYPRE_BigInt size = n_broken;
      Array<HYPRE_BigInt> flux_offsets;
      Array<HYPRE_BigInt> *partitions[] = { &flux_offsets };
      fes.GetParMesh()->GenerateOffsets(1, &size, partitions);
      const HYPRE_BigInt num_global_flux_dofs = flux_offsets[2];

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
      // The gather map may extend past the local DOFs (face neighbors).
      MFEM_VERIFY(interior.Size() >= trace_size, "Unexpected trace gather map.");

      // Keep all interface true DOFs. Owned multipliers on non-interior
      // (boundary) faces are inactive, and their equations are replaced with
      // identity.
      {
         Vector boundary(trace_size), owned_boundary(trace_fes.GetTrueVSize());
         boundary.UseDevice(true);
         owned_boundary.UseDevice(true);
         FlagNegative(interior, trace_size, boundary);
         trace_fes.GetRestrictionOperator()->Mult(boundary, owned_boundary);
         FlaggedIndices(owned_boundary, inactive_trace);
      }

      SparseMatrix local_Ct;
      // Copy the DOF map for device use: the element reads its own on the host.
      const Array<int> dof_map(GetDofMap(fes));
      FormLocalConstraintTranspose(dofs, dof_map, interior, trace_size,
                                   local_Ct);
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

   /// Map conforming essential true DOFs to broken DOF indices.
   void GetBrokenEssentialDofs(const Array<int> &ess_tdofs,
                               Array<int> &ess_broken_dofs) const
   {
      Vector essential(fes.GetTrueVSize()), broken_essential;
      essential.UseDevice(true);
      broken_essential.UseDevice(true);
      essential = 0.0;
      essential.SetSubVector(ess_tdofs, 1.0);
      Prolongate(essential, broken_essential);
      FlaggedIndices(broken_essential, ess_broken_dofs);
   }

   void Assemble(const Vector &broken, Vector &x) const
   {
      Vector conforming(fes.GetVSize());
      conforming.UseDevice(true);
      restriction.MultTranspose(broken, conforming);
      x.SetSize(fes.GetTrueVSize());
      prolongation.MultTranspose(conforming, x);
   }

   /// Assemble the true DOF diagonal of a conforming operator from the
   /// diagonals of its element matrices (as broken DOFs), as for
   /// ParBilinearForm::AssembleDiagonal with partial assembly.
   void AssembleDiagonal(const Vector &broken_diag, Vector &diag) const
   {
      Vector conforming(fes.GetVSize());
      conforming.UseDevice(true);
      restriction.AbsMultTranspose(broken_diag, conforming);
      diag.SetSize(fes.GetTrueVSize());
      prolongation.MultTranspose(conforming, diag);
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

/// Assemble W^{-1} + D M_g^{-1} D^T, with diagonal RT and L2 masses. The RT
/// mass diagonal is given (as true DOFs), and the L2 mass is weighted by beta
/// (one when null). Essential RT columns are eliminated; the caller owns the
/// returned matrix.
inline HypreParMatrix *AssembleGraphLaplacian(
   ParFiniteElementSpace &fes_rt, ParFiniteElementSpace &fes_l2,
   const Array<int> &ess_rt_dofs, const Vector &rt_mass_diag,
   Coefficient *beta = nullptr)
{
   Vector rt_mass_diag_inv(rt_mass_diag);
   rt_mass_diag_inv.UseDevice(true);
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

   std::unique_ptr<HypreParMatrix> D(
      FormDiscreteDivergenceMatrix(fes_rt, fes_l2, ess_rt_dofs));
   // D M^{-1} D^T as a single product, since M is diagonal.
   std::unique_ptr<HypreParMatrix> Minv_Dt(D->Transpose());
   Minv_Dt->ScaleRows(rt_mass_diag_inv);
   HypreParMatrix *S = ParMult(D.get(), Minv_Dt.get(), true);

   // Add W^{-1} to the diagonal in place. Each row of the local diagonal block
   // holds its diagonal entry unless the row of D is empty (all faces of the
   // L2 cell essential); in that case, add W^{-1} as a matrix instead.
   S->HypreReadWrite();
   hypre_ParCSRMatrix *S_hypre = *S;
   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S_hypre);
   const int n = hypre_CSRMatrixNumRows(S_diag);
   const auto I = S_diag->i;
   const auto J = S_diag->j;
   auto data = S_diag->data;
   Vector has_diag(n);
   has_diag.UseDevice(true);
   {
      auto d_has_diag = has_diag.GetMemory().Write(GetHypreMemoryClass(), n);
      mfem::hypre_forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         d_has_diag[i] = 0.0;
         for (int jj = I[i]; jj < I[i+1]; ++jj)
         {
            if (J[jj] == i) { d_has_diag[i] = 1.0; break; }
         }
      });
   }
   int all_diag = (n == 0 || has_diag.Min() > 0.0) ? 1 : 0;
   MPI_Allreduce(MPI_IN_PLACE, &all_diag, 1, MPI_INT, MPI_MIN, S->GetComm());
   if (all_diag)
   {
      const auto w = l2_mass_diag_inv.GetMemory().Read(GetHypreMemoryClass(),
                                                       n);
      mfem::hypre_forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         for (int jj = I[i]; jj < I[i+1]; ++jj)
         {
            if (J[jj] == i) { data[jj] += w[i]; break; }
         }
      });
   }
   else
   {
      auto l2_diag = DiagonalSparseMatrix(l2_mass_diag_inv);
      HypreParMatrix W_inverse(fes_l2.GetComm(), fes_l2.GlobalTrueVSize(),
                               fes_l2.GetTrueDofOffsets(), l2_diag.get());
      HypreParMatrix *S_sum = ParAdd(S, &W_inverse);
      delete S;
      S = S_sum;
   }
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
                       GenericCoefficient alpha = GenericCoefficient(),
                       Coefficient *beta = nullptr, SolverType stype=AMG)
      : Solver(fes_hat.GetNumTraceDofs()), F_inverse(height)
   {
      ParFiniteElementSpace &fes_rt = fes_hat.GetFESpace();
      F_inverse.UseDevice(true);

      // Compute the diagonal of the alpha-weighted RT mass matrix using PA
      Vector diagonal_lex(fes_hat.GetNumBrokenDofs());
      diagonal_lex.UseDevice(true);
      diagonal_lex = 0.0;
      std::unique_ptr<VectorFEMassIntegrator> mass(
         alpha.NewIntegrator<VectorFEMassIntegrator>());
      mass->SetIntRule(GetCoefficientRule(alpha.Q));
      mass->AssemblePA(fes_rt);
      mass->AssembleDiagonalPA(diagonal_lex);
      // The conforming diagonal, for the graph Laplacian.
      Vector rt_mass_diag;
      rt_mass_diag.UseDevice(true);
      fes_hat.AssembleDiagonal(diagonal_lex, rt_mass_diag);

      // The PA diagonal is in the lexicographic broken coordinates.
      Vector &inverse_mass = diagonal_lex;
      inverse_mass.Reciprocal();

      // F = C M^{-1} C^T is diagonal: each broken flux DOF couples to at most
      // one multiplier, with entry sign(dof_map) (see BrokenSpace). Weighting
      // by these signs makes C sum |C| M^{-1}.
      {
         // Copy the DOF map for device use: the element reads its own on the
         // host.
         const Array<int> dof_map(GetDofMap(fes_rt));
         Vector signed_inverse_mass;
         signed_inverse_mass.UseDevice(true);
         MultiplyDofMapSigns(dof_map, inverse_mass, signed_inverse_mass);
         fes_hat.GetC().Mult(signed_inverse_mass, F_inverse);
      }
      // Inactive multipliers have identity equations in the constrained system.
      F_inverse.SetSubVector(fes_hat.GetInactiveTraceDofs(), 1.0);
      F_inverse.Reciprocal();

      const int dim = fes_rt.GetMesh()->Dimension();
      const int order = fes_rt.GetMaxElementOrder();
      Array<int> element2face;
      FormElementToFace(dim, order, element2face);
      L2_FECollection scalar_fec(order-1, dim, BasisType::IntegratedGLL,
                                 FiniteElement::INTEGRAL);
      ParFiniteElementSpace scalar_fes(fes_rt.GetParMesh(), &scalar_fec);

      // Form F^{-1} E^T = F^{-1} C (D_hat M_b^{-1})^T (multipliers by L2 subcells).
      SparseMatrix D_local;
      FormBrokenDivergence(element2face, fes_hat.GetNumDofsPerElem(), dim,
                           fes_rt.GetNE(), inverse_mass, D_local);
      HypreParMatrix &Ct = fes_hat.GetCt();
      HypreParMatrix D_par(fes_rt.GetComm(), scalar_fes.GlobalVSize(),
                           Ct.GetGlobalNumRows(), scalar_fes.GetDofOffsets(),
                           Ct.GetRowStarts(), &D_local);
      std::unique_ptr<HypreParMatrix> Dt(D_par.Transpose());
      F_inverse_Et.reset(ParMult(&fes_hat.GetC(), Dt.get(), true));
      F_inverse_Et->ScaleRows(F_inverse);

      S.reset(AssembleGraphLaplacian(fes_rt, scalar_fes, ess_rt_dofs,
                                     rt_mass_diag, beta));

      if (stype == AMG)
      {
         auto *amg = new HypreBoomerAMG(*S);
         // Printing the setup statistics is slow, see
         // https://github.com/hypre-space/hypre/issues/1288
         amg->SetPrintLevel(0);
         {
            // Set up AMG now, rather than on its first application.
            Vector b(S->Height()), x(S->Height());
            b.UseDevice(true);
            x.UseDevice(true);
            b = 0.0;
            x = 0.0;
            amg->Setup(b, x);
         }
         S_inv.reset(amg);
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
      S_inv->Mult(z1, z2); // overwrites z2 (zero initial guess)
      MultiplyEntries(F_inverse, x, y);
      F_inverse_Et->AddMult(z2, y);
   }
};

/// Apply element inverses in the lexicographic broken coordinates of
/// BrokenSpace. The native element matrices of the form are converted to these
/// coordinates. Essential DOFs use the same broken numbering and are
/// constrained before LU.
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
      const Array<int> &dof_map = GetDofMap(*a.FESpace());
      DenseMatrix lex;
      for (int e = 0; e < lu.SizeK(); ++e)
      {
         DenseMatrix native(lu.GetData(e), nd, nd);
         NativeToLexicographic(dof_map, native, lex);
         native = lex;
      }
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
      // The hybridization constraint has native element rows; convert them to
      // the lexicographic broken coordinates of BrokenSpace.
      SparseMatrix &local_Ct = hybridization.GetConstraintTranspose();
      auto local_Ct_lex = NativeToLexicographicRows(GetDofMap(fes), local_Ct);
      HYPRE_BigInt size = local_Ct.Height();
      Array<HYPRE_BigInt> flux_offsets;
      Array<HYPRE_BigInt> *partitions[] = { &flux_offsets };
      fes.GetParMesh()->GenerateOffsets(1, &size, partitions);
      HypreParMatrix parallel_Ct(fes.GetComm(), flux_offsets.Last(),
                                 trace_fes.GlobalVSize(), flux_offsets.GetData(),
                                 trace_fes.GetDofOffsets(), local_Ct_lex.get());
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

/// Check the structure of the constraint transpose Ct that the hybridized
/// saddle-point operator relies on: the constraint couples only interior
/// faces, so Ct has no entries in the rows of the essential (boundary) broken
/// DOFs, and none in the columns of the inactive (boundary) multipliers.
inline void VerifyConstraintStructure(const HypreParMatrix &Ct,
                                      const Array<int> &ess_broken_dofs,
                                      const Array<int> &inactive_dofs)
{
   Vector a(Ct.Width()), b(Ct.Height()), sub;
   a.UseDevice(true);
   b.UseDevice(true);
   a.Randomize(1);
   Ct.Mult(a, b);
   b.GetSubVector(ess_broken_dofs, sub);
   MFEM_VERIFY(sub.Normlinf() == 0.0,
               "The constraint couples essential broken DOFs.");
   b.Randomize(2);
   Ct.MultTranspose(b, a);
   a.GetSubVector(inactive_dofs, sub);
   MFEM_VERIFY(sub.Normlinf() == 0.0,
               "The constraint couples inactive multipliers.");
}

/// The constrained hybridized saddle-point operator
///
///    [ A  C^T ]
///    [ C  I_i ]
///
/// in broken flux and multiplier coordinates, where I_i is the identity on the
/// inactive multipliers. A must already be constrained (identity rows and
/// columns for the essential broken DOFs), and Ct must have the structure
/// checked by VerifyConstraintStructure, so that the essential and inactive
/// equations are identities. Each output block is written directly, without
/// the temporaries of a BlockOperator or the copies of a ConstrainedOperator.
class HybridSaddlePointOperator : public Operator
{
   const Operator &A;
   const HypreParMatrix &Ct;
   const Array<int> &inactive_dofs;
   mutable Vector x0, x1, y0, y1;

public:
   HybridSaddlePointOperator(const Operator &A_, const HypreParMatrix &Ct_,
                             const Array<int> &inactive_dofs_)
      : Operator(A_.Height() + Ct_.Width()), A(A_), Ct(Ct_),
        inactive_dofs(inactive_dofs_) { }

   void Mult(const Vector &x, Vector &y) const override
   {
      const int n0 = A.Height(), n1 = Ct.Width();
      x.Read();
      y.Write();
      x0.MakeRef(const_cast<Vector&>(x), 0, n0);
      x1.MakeRef(const_cast<Vector&>(x), n0, n1);
      y0.MakeRef(y, 0, n0);
      y1.MakeRef(y, n0, n1);
      A.Mult(x0, y0);
      Ct.Mult(1.0, x1, 1.0, y0);
      Ct.MultTranspose(x0, y1);
      CopyEntries(inactive_dofs, x1, y1);
      y0.SyncAliasMemory(y);
      y1.SyncAliasMemory(y);
   }
};

/// The multiplier Schur complement C A^{-1} C^T + I_i of the hybridized
/// saddle-point operator, where I_i is the identity on the inactive
/// multipliers.
/** A_inv must be the exact inverse of the constrained A (no inner iterations
    are performed), e.g. FastDiagonalizationSolver on affine orthogonal meshes
    with elementwise constant coefficients. Ct must have the structure checked
    by VerifyConstraintStructure. The operator is then SPD, and the hybridized
    system can be solved by CG on the multipliers. */
class HybridSchurComplementOperator : public Operator
{
   const Solver &A_inv;
   const HypreParMatrix &Ct;
   const Array<int> &inactive_dofs;
   mutable Vector z, w;

public:
   HybridSchurComplementOperator(const Solver &A_inv_, const HypreParMatrix &Ct_,
                                 const Array<int> &inactive_dofs_)
      : Operator(Ct_.Width()), A_inv(A_inv_), Ct(Ct_),
        inactive_dofs(inactive_dofs_), z(Ct_.Height()), w(Ct_.Height())
   {
      z.UseDevice(true);
      w.UseDevice(true);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      Ct.Mult(x, z);
      A_inv.Mult(z, w);
      Ct.MultTranspose(w, y);
      CopyEntries(inactive_dofs, x, y);
   }
};

} // namespace mfem

#endif // MFEM_HDIV_HB_HPP
