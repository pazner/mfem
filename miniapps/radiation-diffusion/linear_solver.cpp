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

#include "radiation_diffusion.hpp"
#include "linear_solver.hpp"
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

RadiationDiffusionLinearSolver::RadiationDiffusionLinearSolver(
   ParMesh &mesh, ParFiniteElementSpace &fes_rt_, ParFiniteElementSpace &fes_l2_)
   : minres(mesh.GetComm()),
     order(fes_rt_.GetMaxElementOrder()),
     fec_l2(order - 1, mesh.Dimension(), b2, mt),
     fes_l2(&mesh, &fec_l2),
     fec_rt(order - 1, mesh.Dimension(), b1, b2),
     fes_rt(&mesh, &fec_rt),
     basis_l2(&fes_l2, &fes_l2_),
     basis_rt(&fes_rt, &fes_rt_),
     mass_rt(&fes_rt),
     dt_prev(0.0)
{
   mass_rt.AddDomainIntegrator(new VectorFEMassIntegrator(&R_coeff));
   mass_rt.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   D.reset(FormDiscreteDivergenceMatrix(fes_rt, fes_l2, ess_dofs));
   Dt.reset(D->Transpose());

   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = fes_l2.GetTrueVSize();
   offsets[2] = offsets[1] + fes_rt.GetTrueVSize();

   minres.SetAbsTol(0.0);
   minres.SetRelTol(1e-12);
   minres.SetMaxIter(500);
   minres.SetPrintLevel(IterativeSolver::PrintLevel().None());

   basis_l2.AddDomainInterpolator(new IdentityInterpolator);
   basis_rt.AddDomainInterpolator(new IdentityInterpolator);

   basis_l2.Assemble();
   basis_rt.Assemble();

   basis_l2.Finalize();
   basis_rt.Finalize();

   B_l2.reset(basis_l2.ParallelAssemble());
   B_rt.reset(basis_rt.ParallelAssemble());

   S_inv.SetPrintLevel(0);
}

void RadiationDiffusionLinearSolver::Setup(const double dt)
{
   using namespace MMS;

   if (dt == dt_prev) { return; }
   dt_prev = dt;

   L_coeff.constant = 1.0 + c*dt*sigma;
   R_coeff.constant = 3.0*sigma/c/dt; // <-- NOTE: sign difference here

   // Reassmble the RT mass operator with the new coefficient
   mass_rt.Assemble();
   mass_rt.FormSystemMatrix(ess_dofs, R);

   // Recreate the DG mass inverse with the new coefficient
   L_inv.reset(new DGMassInverse(fes_l2, L_coeff));
   // L_inv.reset(new DGMassInverse_Direct(fes_l2, L_coeff));

   // Form the updated approximate Schur complement
   Vector R_diag(fes_rt.GetTrueVSize());
   mass_rt.AssembleDiagonal(R_diag);
   std::unique_ptr<HypreParMatrix> R_diag_inv(DiagonalInverse(R_diag, fes_rt));

   Vector L_diag(fes_l2.GetTrueVSize());
   ParBilinearForm mass_l2(&fes_l2);
   mass_l2.AddDomainIntegrator(new MassIntegrator(L_coeff));
   mass_l2.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass_l2.Assemble();
   mass_l2.AssembleDiagonal(L_diag);
   std::unique_ptr<HypreParMatrix> L_diag_inv(DiagonalInverse(L_diag, fes_l2));

   std::unique_ptr<HypreParMatrix> D_Minv_Dt(RAP(R_diag_inv.get(), Dt.get()));
   S.reset(ParAdd(D_Minv_Dt.get(), L_diag_inv.get()));

   // Reassemble the preconditioners
   R_inv.SetOperator(mass_rt);
   S_inv.SetOperator(*S);

   // Set up the block operators
   A_block.reset(new BlockOperator(offsets));
   A_block->SetBlock(0, 0, L_inv.get());
   A_block->SetBlock(0, 1, D.get());
   A_block->SetBlock(1, 0, Dt.get());
   A_block->SetBlock(1, 1, R.Ptr(), -1.0);

   D_prec.reset(new BlockDiagonalPreconditioner(offsets));
   D_prec->SetDiagonalBlock(0, &S_inv);
   D_prec->SetDiagonalBlock(1, &R_inv);

   minres.SetPreconditioner(*D_prec);
   minres.SetOperator(*A_block);
}

void RadiationDiffusionLinearSolver::Mult(const Vector &b, Vector &x) const
{
   b_prime.SetSize(b.Size());
   x_prime.SetSize(x.Size());

   // Transform RHS
   Vector bE_prime(b_prime, offsets[0], offsets[1]-offsets[0]);
   Vector bF_prime(b_prime, offsets[1], offsets[2]-offsets[1]);

   const Vector bE(const_cast<Vector&>(b), offsets[0], offsets[1]-offsets[0]);
   const Vector bF(const_cast<Vector&>(b), offsets[1], offsets[2]-offsets[1]);

   z.SetSize(bE.Size());
   L_inv->Mult(bE, z);

   B_l2->MultTranspose(z, bE_prime);
   B_rt->MultTranspose(bF, bF_prime);

   // Solve the transformed system
   x_prime = 0.0; // TODO: minres.iterative_mode = false?
   minres.Mult(b_prime, x_prime);

   // Transform the solution
   Vector xE_prime(x_prime, offsets[0], offsets[1]-offsets[0]);
   Vector xF_prime(x_prime, offsets[1], offsets[2]-offsets[1]);

   Vector xE(x, offsets[0], offsets[1]-offsets[0]);
   Vector xF(x, offsets[1], offsets[2]-offsets[1]);

   L_inv->Mult(xE_prime, z);

   B_l2->Mult(z, xE);
   B_rt->Mult(xF_prime, xF);
}

void RadiationDiffusionLinearSolver::SetOperator(const Operator &op) { }

} // namespace mfem
