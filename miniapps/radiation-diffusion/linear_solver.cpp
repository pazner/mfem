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

HypreParMatrix *DiagonalInverse(Vector &diag_vec,
                                const ParFiniteElementSpace &fes)
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
   RadiationDiffusionOperator &rad_diff_)
   : Solver(rad_diff_.Height()),
     rad_diff(rad_diff_),
     minres(rad_diff.GetComm()),
     fec_l2(rad_diff.fec_l2.GetOrder(), rad_diff.dim, b2, mt),
     fes_l2(rad_diff.fes_l2.GetParMesh(), &fec_l2),
     fec_rt(rad_diff.fec_l2.GetOrder(), rad_diff.dim, b1, b2),
     fes_rt(rad_diff.fes_l2.GetParMesh(), &fec_rt),
     mass_rt(&fes_rt),
     dt_prev(0.0)
{
   mass_rt.AddDomainIntegrator(new VectorFEMassIntegrator(&R_coeff));
   mass_rt.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   D.reset(FormDiscreteDivergenceMatrix(fes_rt, fes_l2, ess_dofs));
   Dt.reset(D->Transpose());

   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = fes_rt.GetTrueVSize();
   offsets[2] = offsets[1] + fes_l2.GetTrueVSize();

   minres.SetAbsTol(0.0);
   minres.SetRelTol(1e-12);
   minres.SetMaxIter(500);
   minres.SetPrintLevel(IterativeSolver::PrintLevel().None());

   S_inv.SetPrintLevel(0);
}

void RadiationDiffusionLinearSolver::Setup()
{
   using namespace MMS;

   const double dt = rad_diff.dt;
   if (dt == dt_prev) { return; }
   dt_prev = dt;

   L_coeff.constant = 1.0 + c*dt*sigma;
   R_coeff.constant = 3.0*sigma/c/dt; // <-- NOTE: sign difference here

   // Reassmble the RT mass operator with the new coefficient
   mass_rt.Assemble();
   mass_rt.FormSystemMatrix(ess_dofs, R);

   // Recreate the DG mass inverse with the new coefficient
   L_inv.reset(new DGMassInverse(fes_l2, L_coeff));

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
   A_block->SetBlock(0, 0, R.Ptr());
   A_block->SetBlock(0, 1, Dt.get());
   A_block->SetBlock(1, 0, D.get());
   A_block->SetBlock(1, 1, L_inv.get(), -1.0);

   D_prec.reset(new BlockDiagonalPreconditioner(offsets));
   D_prec->SetDiagonalBlock(0, &R_inv);
   D_prec->SetDiagonalBlock(1, &S_inv);

   minres.SetOperator(*A_block);
   minres.SetPreconditioner(*D_prec);
}

void RadiationDiffusionLinearSolver::Mult(const Vector &b, Vector &x) const
{
   minres.Mult(b, x);
}

void RadiationDiffusionLinearSolver::SetOperator(const Operator &op) { }

} // namespace mfem
