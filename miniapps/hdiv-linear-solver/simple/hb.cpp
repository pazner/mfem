
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

#include "hb.hpp"
#include "general/scan.hpp"

namespace mfem
{

namespace
{

inline MFEM_HOST_DEVICE real_t Sign(int d) { return d < 0 ? -1.0 : 1.0; }

std::unique_ptr<SparseMatrix> DiagonalSparseMatrix(const Vector &d)
{
   const int n = d.Size();
   auto D = std::make_unique<SparseMatrix>(n, n, 1);
   const real_t *d_d = d.Read();
   int *J = D->WriteJ();
   real_t *A = D->WriteData();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { J[i] = i; A[i] = d_d[i]; });
   return D;
}

/// Add to existing diagonal entries, or return false without modifying A.
bool TryAddDiagonal(HypreParMatrix &A, const Vector &diagonal)
{
   A.HypreReadWrite();
   hypre_ParCSRMatrix *parallel = A;
   const auto block = hypre_ParCSRMatrixDiag(parallel);
   const int n = A.Height();
   const auto I = hypre_CSRMatrixI(block);
   const auto J = hypre_CSRMatrixJ(block);
   auto data = hypre_CSRMatrixData(block);
   Vector found(n);
   found.UseDevice(true);
   auto flags = found.GetMemory().Write(GetHypreMemoryClass(), n);
   mfem::hypre_forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      flags[i] = 0.0;
      for (int j = I[i]; j < I[i+1]; ++j)
      {
         if (J[j] == i) { flags[i] = 1.0; break; }
      }
   });
   int complete = (n == 0 || found.Min() > 0.0);
   // All ranks must take the same branch: the fallback uses parallel addition.
   MPI_Allreduce(MPI_IN_PLACE, &complete, 1, MPI_INT, MPI_MIN, A.GetComm());
   if (!complete) { return false; }

   const auto d = diagonal.GetMemory().Read(GetHypreMemoryClass(), n);
   mfem::hypre_forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      for (int j = I[i]; j < I[i+1]; ++j)
      {
         if (J[j] == i) { data[j] += d[i]; break; }
      }
   });
   return true;
}

const Array<int> &GetDofMap(const FiniteElementSpace &fes)
{
   return dynamic_cast<const TensorBasisElement&>(*fes.GetTypicalFE())
          .GetDofMap();
}

void MultiplyDofMapSigns(const Array<int> &dof_map, const Vector &x,
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

void FlaggedIndices(const Vector &flags, Array<int> &idx)
{
   const int n = flags.Size();
   Array<int> iota(n), count(1);
   int *d_iota = iota.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { d_iota[i] = i; });
   idx.SetSize(n);
   CopyFlagged(true, iota.Read(), flags.Read(), idx.Write(), count.Write(), n);
   idx.SetSize(count.HostRead()[0]);
}

void InvertCounts(Vector &counts)
{
   real_t *d_c = counts.ReadWrite();
   mfem::forall(counts.Size(), [=] MFEM_HOST_DEVICE (int i)
   { d_c[i] = d_c[i] > 0.0 ? 1.0/d_c[i] : 1.0; });
}

void FlagNegative(const Array<int> &map, int n, Vector &flags)
{
   flags.SetSize(n);
   const int *d_map = map.Read();
   real_t *d_flags = flags.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   { d_flags[i] = d_map[i] < 0 ? 1.0 : 0.0; });
}

void FormLocalConstraintTranspose(const Array<int> &dofs,
                                  const Array<int> &dof_map,
                                  const Array<int> &interior,
                                  int trace_size, SparseMatrix &Ct)
{
   const int n = dofs.Size();
   const int nd = dof_map.Size();
   Ct.OverrideSize(n, trace_size);
   Ct.GetMemoryI().New(n + 1);
   Array<int> nnz(1);
   {
      const int *d_dofs = dofs.Read();
      const int *d_interior = interior.Read();
      int *I = Ct.WriteI();
      mfem::forall(n + 1, [=] MFEM_HOST_DEVICE (int i)
      {
         if (i == n) { I[i] = 0; return; }
         const int td = UnsignIndex(d_dofs[i]);
         I[i] = (td < trace_size && d_interior[td] >= 0) ? 1 : 0;
      });
      ExclusiveScan(true, I, I, n + 1, 0);
      int *d_nnz = nnz.Write();
      mfem::forall(1, [=] MFEM_HOST_DEVICE (int) { d_nnz[0] = I[n]; });
   }
   const int local_nnz = nnz.HostRead()[0];
   Ct.GetMemoryJ().New(local_nnz);
   Ct.GetMemoryData().New(local_nnz);
   const int *I = Ct.ReadI();
   const int *d_dofs = dofs.Read();
   const int *map = dof_map.Read();
   int *J = Ct.WriteJ();
   real_t *A = Ct.WriteData();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      if (I[i+1] > I[i])
      {
         J[I[i]] = UnsignIndex(d_dofs[i]);
         A[I[i]] = Sign(map[i%nd]);
      }
   });
}

void FormBrokenDivergence(const Array<int> &element2face, int nd,
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

/// Signed incidence of each subcell on its 2*dim RT flux DOFs.
void FormElementToFace(int dim, int p, Array<int> &element2face)
{
   const int nvol = dim == 2 ? p*p : p*p*p;
   const int nc = (p+1)*nvol/p;
   element2face.SetSize(2*dim*nvol);
   for (int i = 0; i < nvol; ++i)
   {
      int stride = 1;
      for (int c = 0; c < dim; ++c)
      {
         const int left = c*nc + i + (i/(stride*p))*stride;
         element2face[2*(c + dim*i)] = -1-left;
         element2face[2*(c + dim*i)+1] = left+stride;
         stride *= p;
      }
   }
}

/// Assemble the topological divergence from conforming RT to integral L2 DOFs.
std::unique_ptr<HypreParMatrix> FormDiscreteDivergence(
   ParFiniteElementSpace &fes_rt, ParFiniteElementSpace &fes_l2)
{
   const int dim = fes_rt.GetMesh()->Dimension();
   const int nd = fes_rt.GetTypicalFE()->GetDof();
   const int nvol = fes_l2.GetTypicalFE()->GetDof();
   const int nrows = fes_l2.GetVSize();
   const int entries = 2*dim;
   Array<int> element2face;
   FormElementToFace(dim, fes_rt.GetMaxElementOrder(), element2face);
   const auto &restriction = dynamic_cast<const ElementRestriction&>(
                                *fes_rt.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC));
   const int *gather = restriction.GatherMap().Read();
   const int *e2f = element2face.Read();

   SparseMatrix local(nrows, fes_rt.GetVSize(), entries);
   int *J = local.WriteJ();
   real_t *V = local.WriteData();
   mfem::forall(nrows*entries, [=] MFEM_HOST_DEVICE (int k)
   {
      const int row = k/entries, e = row/nvol;
      const int face = e2f[k%(entries*nvol)];
      const int dof = gather[e*nd + UnsignIndex(face)];
      J[k] = UnsignIndex(dof);
      V[k] = Sign(face)*Sign(dof);
   });

   HypreParMatrix parallel(fes_rt.GetComm(), fes_l2.GlobalVSize(),
                           fes_rt.GlobalVSize(), fes_l2.GetDofOffsets(),
                           fes_rt.GetDofOffsets(), &local);
   auto D = std::unique_ptr<HypreParMatrix>(
               ParMult(&parallel, fes_rt.Dof_TrueDof_Matrix(), true));
   // The integral L2 space is discontinuous, so its prolongation is identity.
   return D;
}

std::unique_ptr<HypreParMatrix> AssembleGraphLaplacian(
   ParFiniteElementSpace &fes_rt, ParFiniteElementSpace &fes_l2,
   const Array<int> &ess_rt_dofs, const Vector &rt_mass_diag, Coefficient &beta)
{
   Vector rt_mass_diag_inv(rt_mass_diag);
   rt_mass_diag_inv.UseDevice(true);
   rt_mass_diag_inv.Reciprocal();
   // Zero weights remove essential flux columns from D M^{-1} D^T.
   rt_mass_diag_inv.SetSubVector(ess_rt_dofs, 0.0);

   ParBilinearForm mass_l2(&fes_l2);
   mass_l2.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass_l2.AddDomainIntegrator(new MassIntegrator(beta));
   mass_l2.Assemble();
   Vector l2_mass_diag_inv(fes_l2.GetTrueVSize());
   l2_mass_diag_inv.UseDevice(true);
   mass_l2.AssembleDiagonal(l2_mass_diag_inv);
   l2_mass_diag_inv.Reciprocal();

   // Form D M^{-1} D^T with a row scaling and one parallel matrix product.
   auto D = FormDiscreteDivergence(fes_rt, fes_l2);
   std::unique_ptr<HypreParMatrix> Dt(D->Transpose());
   Dt->ScaleRows(rt_mass_diag_inv);
   auto S = std::unique_ptr<HypreParMatrix>(ParMult(D.get(), Dt.get(), true));
   // Normally every row has a structural diagonal, even when its weight is zero.
   if (!TryAddDiagonal(*S, l2_mass_diag_inv))
   {
      auto l2_diag = DiagonalSparseMatrix(l2_mass_diag_inv);
      HypreParMatrix W_inverse(fes_l2.GetComm(), fes_l2.GlobalTrueVSize(),
                               fes_l2.GetTrueDofOffsets(), l2_diag.get());
      S.reset(ParAdd(S.get(), &W_inverse));
   }
   S->CopyRowStarts();
   S->CopyColStarts();
   return S;
}

} // namespace

BrokenSpace::BrokenSpace(ParFiniteElementSpace &fes_)
   : fes(fes_), trace_fec(fes.GetMaxElementOrder()-1, fes.GetMesh()->Dimension()),
     trace_fes(fes.GetParMesh(), &trace_fec),
     restriction(dynamic_cast<const ElementRestriction&>(
                    *fes.GetElementRestriction(
                       ElementDofOrdering::LEXICOGRAPHIC))),
     prolongation(*fes.GetProlongationMatrix())

{
   const auto &dofs = restriction.GatherMap();
   const auto &trace_prolongation = *trace_fes.Dof_TrueDof_Matrix();
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

int BrokenSpace::GetNumBrokenDofs() const
{
   return restriction.Height();
}

ParFiniteElementSpace &BrokenSpace::GetFESpace() const { return fes; }

int BrokenSpace::GetNumTraceDofs() const
{
   return trace_fes.GetTrueVSize();
}

HypreParMatrix &BrokenSpace::GetC()
{
   return *C;
}

HypreParMatrix &BrokenSpace::GetCt()
{
   return *Ct;
}

const Array<int> &BrokenSpace::GetInactiveTraceDofs() const
{
   return inactive_trace;
}

void BrokenSpace::LocalProlongate(const Vector &x, Vector &broken) const
{
   broken.SetSize(restriction.Height());
   restriction.Mult(x, broken);
}

void BrokenSpace::Prolongate(const Vector &x, Vector &broken) const
{
   Vector conforming(fes.GetVSize());
   conforming.UseDevice(true);
   prolongation.Mult(x, conforming);
   LocalProlongate(conforming, broken);
}

void BrokenSpace::GetBrokenEssentialDofs(const Array<int> &ess_tdofs,
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

void BrokenSpace::AverageTranspose(const Vector &x, Vector &broken) const
{
   Vector scaled(x);
   scaled.UseDevice(true);
   scaled *= inverse_multiplicity;
   Prolongate(scaled, broken);
}

void BrokenSpace::Average(const Vector &broken, Vector &x) const
{
   Vector conforming(fes.GetVSize());
   conforming.UseDevice(true);
   restriction.MultTranspose(broken, conforming);
   x.SetSize(fes.GetTrueVSize());
   prolongation.MultTranspose(conforming, x);
   x *= inverse_multiplicity;
}

void BrokenSpace::AssembleDiagonal(const Vector &broken_diag,
                                   Vector &diag) const
{
   Vector local(fes.GetVSize());
   local.UseDevice(true);
   restriction.AbsMultTranspose(broken_diag, local);
   diag.SetSize(fes.GetTrueVSize());
   prolongation.MultTranspose(local, diag);
}

GraphPreconditioner::GraphPreconditioner(BrokenSpace &fes_hat,
                                         const Array<int> &ess_rt_dofs,
                                         Coefficient &alpha, Coefficient &beta)
   : Solver(fes_hat.GetNumTraceDofs()), F_inverse(height)
{
   ParFiniteElementSpace &fes_rt = fes_hat.GetFESpace();
   F_inverse.UseDevice(true);

   // Compute the diagonal of the alpha-weighted RT mass matrix using PA
   Vector inverse_mass(fes_hat.GetNumBrokenDofs());
   inverse_mass.UseDevice(true);
   inverse_mass = 0.0;
   VectorFEMassIntegrator mass(alpha);
   mass.AssemblePA(fes_rt);
   mass.AssembleDiagonalPA(inverse_mass);
   // Assemble the conforming mass before inverting the element diagonals.
   Vector rt_mass_diag;
   rt_mass_diag.UseDevice(true);
   fes_hat.AssembleDiagonal(inverse_mass, rt_mass_diag);
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
   FormBrokenDivergence(element2face, fes_rt.GetTypicalFE()->GetDof(), dim,
                        fes_rt.GetNE(), inverse_mass, D_local);
   HypreParMatrix &Ct = fes_hat.GetCt();
   HypreParMatrix D_par(fes_rt.GetComm(), scalar_fes.GlobalVSize(),
                        Ct.GetGlobalNumRows(), scalar_fes.GetDofOffsets(),
                        Ct.GetRowStarts(), &D_local);
   std::unique_ptr<HypreParMatrix> Dt(D_par.Transpose());
   F_inverse_Et.reset(ParMult(&fes_hat.GetC(), Dt.get(), true));
   F_inverse_Et->ScaleRows(F_inverse);

   S = AssembleGraphLaplacian(fes_rt, scalar_fes, ess_rt_dofs, rt_mass_diag, beta);

   S_inv = std::make_unique<HypreBoomerAMG>(*S);
   S_inv->SetPrintLevel(0);
   z1.SetSize(S->Height());
   z2.SetSize(S->Height());
   z1.UseDevice(true);
   z2.UseDevice(true);
   // Charge AMG setup to construction, not the first MINRES iteration.
   z1 = 0.0;
   z2 = 0.0;
   S_inv->Setup(z1, z2);
}

void GraphPreconditioner::SetOperator(const Operator &)
{
   MFEM_ABORT("GraphPreconditioner must be reconstructed.");
}

void GraphPreconditioner::Mult(const Vector &x, Vector &y) const
{
   F_inverse_Et->MultTranspose(x, z1);
   S_inv->Mult(z1, z2); // overwrites z2 (zero initial guess)
   y = x;
   y *= F_inverse;
   F_inverse_Et->AddMult(z2, y);
}

} // namespace mfem
