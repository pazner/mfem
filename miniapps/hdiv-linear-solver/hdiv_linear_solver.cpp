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

#include "general/forall.hpp"
#include "hdiv_linear_solver.hpp"
#include "discrete_divergence.hpp"

namespace mfem
{

HypreParMatrix *DiagonalInverse(
   Vector &diag_vec, const ParFiniteElementSpace &fes)
{
   diag_vec.HostReadWrite();
   for (int i=0; i<diag_vec.Size(); ++i) { diag_vec[i] = 1.0/diag_vec[i]; }
   SparseMatrix diag_inv(diag_vec);

   HYPRE_BigInt global_size = fes.GlobalTrueVSize();
   HYPRE_BigInt *row_starts = fes.GetTrueDofOffsets();
   HypreParMatrix D(MPI_COMM_WORLD, global_size, row_starts, &diag_inv);
   return new HypreParMatrix(D); // make a deep copy
}

const IntegrationRule &GetMassIntRule(FiniteElementSpace &fes_l2)
{
   Mesh *mesh = fes_l2.GetMesh();
   const FiniteElement *fe = fes_l2.GetFE(0);
   return MassIntegrator::GetRule(*fe, *fe, *mesh->GetElementTransformation(0));
}

HdivSaddlePointLinearSolver::HdivSaddlePointLinearSolver(
   ParMesh &mesh, ParFiniteElementSpace &fes_rt_, ParFiniteElementSpace &fes_l2_,
   Coefficient &L_coeff_, Coefficient &R_coeff_, const Array<int> &ess_rt_dofs_,
   L2CoefficientMode coeff_mode_)
   : minres(mesh.GetComm()),
     order(fes_rt_.GetMaxElementOrder()),
     fec_l2(order - 1, mesh.Dimension(), b2, mt),
     fes_l2(&mesh, &fec_l2),
     fec_rt(order - 1, mesh.Dimension(), b1, b2),
     fes_rt(&mesh, &fec_rt),
     ess_rt_dofs(ess_rt_dofs_),
     basis_l2(fes_l2_, fes_l2),
     basis_rt(fes_rt_, fes_rt),
     mass_l2(&fes_l2),
     mass_rt(&fes_rt),
     L_coeff(L_coeff_),
     R_coeff(R_coeff_),
     coeff_mode(coeff_mode_),
     qs(mesh, GetMassIntRule(fes_l2)),
     qf(qs),
     L_inv_coeff(qf)
{
   // Verify that "zero" coefficient mode matches zero coefficient:
   if (coeff_mode == L2CoefficientMode::ZERO)
   {
      auto *L_const_coeff = dynamic_cast<ConstantCoefficient*>(&L_coeff);
      if (!L_const_coeff || L_const_coeff->constant != 0.0)
      {
         MFEM_ABORT("L2CoefficientMode::ZERO requires zero L coefficient.");
      }
   }

   mass_l2.AddDomainIntegrator(new MassIntegrator(L_inv_coeff));
   mass_l2.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   mass_rt.AddDomainIntegrator(new VectorFEMassIntegrator(&R_coeff));
   mass_rt.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   D.reset(FormDiscreteDivergenceMatrix(fes_rt, fes_l2, ess_rt_dofs));
   Dt.reset(D->Transpose());

   // Versions without BCs needed for elimination
   D_e.reset(FormDiscreteDivergenceMatrix(fes_rt, fes_l2, empty));
   mass_rt.FormSystemMatrix(empty, R_e);

   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = fes_l2.GetTrueVSize();
   offsets[2] = offsets[1] + fes_rt.GetTrueVSize();

   minres.SetAbsTol(0.0);
   minres.SetRelTol(1e-12);
   minres.SetMaxIter(500);
   minres.SetPrintLevel(IterativeSolver::PrintLevel().None());
   minres.iterative_mode = false;

   R_diag.SetSize(fes_rt.GetTrueVSize());
   L_diag.SetSize(fes_l2.GetTrueVSize());

   S_inv.SetPrintLevel(0);

   Setup();
}

HdivSaddlePointLinearSolver::HdivSaddlePointLinearSolver(
   ParMesh &mesh, ParFiniteElementSpace &fes_rt_, ParFiniteElementSpace &fes_l2_,
   Coefficient &R_coeff_, const Array<int> &ess_rt_dofs_)
   : HdivSaddlePointLinearSolver(mesh, fes_rt_, fes_l2_, zero, R_coeff_, ess_rt_dofs_, L2CoefficientMode::ZERO)
{ }

void HdivSaddlePointLinearSolver::Setup()
{
   if (coeff_mode == L2CoefficientMode::IDENTITY)
   {
      L_coeff.Project(qf);
   }
   else if (coeff_mode == L2CoefficientMode::RECIPROCAL)
   {
      L_coeff.Project(qf);
      // Compute L_inv_coeff, which is the reciprocal of L_inv.
      // The data is stored in the QuadratureFunction qf.
      double *qf_d = qf.ReadWrite();
      MFEM_FORALL(i, qf.Size(), {
         qf_d[i] = 1.0/qf_d[i];
      });
   }
   else if (coeff_mode == L2CoefficientMode::ZERO)
   {
      double *qf_d = qf.ReadWrite();
      MFEM_FORALL(i, qf.Size(), qf_d[i] = 1.0;);
   }

   // Form the DG mass inverse with the new coefficient
   L_inv.reset(new DGMassInverse(fes_l2, L_inv_coeff));

   // Reassemble the L2 mass diagonal with the new coefficient
   mass_l2.Assemble();
   mass_l2.AssembleDiagonal(L_diag);
   mass_l2.FormSystemMatrix(empty, L);

   // Reassmble the RT mass operator with the new coefficient
   mass_rt.Update();
   mass_rt.Assemble();
   mass_rt.FormSystemMatrix(ess_rt_dofs, R);

   // Form the updated approximate Schur complement
   mass_rt.AssembleDiagonal(R_diag);

   {
      const int *d_I = ess_rt_dofs.Read();
      double *d_R_diag = R_diag.ReadWrite();
      MFEM_FORALL(i, ess_rt_dofs.Size(), d_R_diag[d_I[i]] = 1.0;);
   }

   std::unique_ptr<HypreParMatrix> R_diag_inv(DiagonalInverse(R_diag, fes_rt));
   std::unique_ptr<HypreParMatrix> D_Minv_Dt(RAP(R_diag_inv.get(), Dt.get()));
   std::unique_ptr<HypreParMatrix> L_diag_inv(DiagonalInverse(L_diag, fes_l2));
   S.reset(ParAdd(D_Minv_Dt.get(), L_diag_inv.get()));

   // Reassemble the preconditioners
   R_inv.reset(new OperatorJacobiSmoother(mass_rt, ess_rt_dofs));
   S_inv.SetOperator(*S);

   // Set up the block operators
   A_block.reset(new BlockOperator(offsets));
   if (coeff_mode != L2CoefficientMode::ZERO)
   {
      A_block->SetBlock(0, 0, L_inv.get());
   }
   A_block->SetBlock(0, 1, D.get());
   A_block->SetBlock(1, 0, Dt.get());
   A_block->SetBlock(1, 1, R.Ptr(), -1.0);

   D_prec.reset(new BlockDiagonalPreconditioner(offsets));
   D_prec->SetDiagonalBlock(0, &S_inv);
   D_prec->SetDiagonalBlock(1, R_inv.get());

   minres.SetPreconditioner(*D_prec);
   minres.SetOperator(*A_block);
}

void HdivSaddlePointLinearSolver::EliminateBC(Vector &b) const
{
   const int n_ess_dofs = ess_rt_dofs.Size();
   if (n_ess_dofs == 0) { return; }

   const int n_l2 = offsets[1];
   const int n_rt = offsets[2]-offsets[1];
   Vector bE(b, 0, n_l2);
   Vector bF(b, n_l2, n_rt);

   MFEM_VERIFY(x_bc.Size() == n_rt, "BCs not set");

   z.SetSize(n_rt);
   z.UseDevice(true);
   z = 0.0;
   const int *d_I = ess_rt_dofs.Read();
   const double *d_x_bc = x_bc.Read();
   double *d_z = z.ReadWrite();
   MFEM_FORALL(i, n_ess_dofs,
   {
      const int j = d_I[i];
      d_z[j] = d_x_bc[j];
   });

   w.SetSize(n_rt);
   basis_rt.MultInverse(z, w);

   // Eliminate the BCs in the L2 RHS
   D_e->Mult(-1.0, w, 1.0, bE);

   // Eliminate the BCs in the RT RHS
   // Flip the sign because the R block appears with multiplier -1
   z.SetSize(n_rt);
   R_e->Mult(w, z);
   bF += z;

   const double *d_w = w.Read();
   double *d_bF = bF.ReadWrite(); // Need read-write access to set subvector
   MFEM_FORALL(i, n_ess_dofs,
   {
      const int j = d_I[i];
      // d_bF[j] = -d_x_bc[j];
      d_bF[j] = -d_w[j];
   });

   bE.SyncAliasMemory(b);
   bF.SyncAliasMemory(b);
}

void HdivSaddlePointLinearSolver::Mult(const Vector &b, Vector &x) const
{
   b_prime.SetSize(b.Size());
   x_prime.SetSize(x.Size());

   // Transform RHS
   Vector bE_prime(b_prime, offsets[0], offsets[1]-offsets[0]);
   Vector bF_prime(b_prime, offsets[1], offsets[2]-offsets[1]);

   const Vector bE(const_cast<Vector&>(b), offsets[0], offsets[1]-offsets[0]);
   const Vector bF(const_cast<Vector&>(b), offsets[1], offsets[2]-offsets[1]);

   z.SetSize(bE.Size());
   basis_l2.MultTranspose(bE, z);

   L_inv->Mult(z, bE_prime);
   basis_rt.MultTranspose(bF, bF_prime);

   // Update the monolithic transformed RHS
   bE_prime.SyncAliasMemory(b_prime);
   bF_prime.SyncAliasMemory(b_prime);

   EliminateBC(b_prime);

   // Solve the transformed system
   minres.Mult(b_prime, x_prime);

   // Transform the solution
   Vector xE_prime(x_prime, offsets[0], offsets[1]-offsets[0]);
   Vector xF_prime(x_prime, offsets[1], offsets[2]-offsets[1]);

   Vector xE(x, offsets[0], offsets[1]-offsets[0]);
   Vector xF(x, offsets[1], offsets[2]-offsets[1]);

   L_inv->Mult(xE_prime, z);

   basis_l2.Mult(z, xE);
   basis_rt.Mult(xF_prime, xF);

   // Update the monolithic solution vector
   xE.SyncAliasMemory(x);
   xF.SyncAliasMemory(x);
}

void HdivSaddlePointLinearSolver::SetOperator(const Operator &op) { }

} // namespace mfem
