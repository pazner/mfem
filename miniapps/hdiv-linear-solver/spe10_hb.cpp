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

#include "mfem.hpp"
#include "hb.hpp"
#include "fast_diag.hpp"
#include "broken_hdiv.hpp"
#include "spe10.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

// Solves the same grad-div problem as spe10.cpp,
//
//    alpha u - grad(beta div u) = 0,
//
// on the SPE10 domain, where alpha is the (diagonal, anisotropic) SPE10
// coefficient and beta is constant, with the normal flux u.n = (1,0,0).n
// prescribed on the whole boundary. The hybridized system (broken fluxes and
// interface multipliers) is solved with MINRES, preconditioned by fast
// diagonalization for the element blocks and the graph preconditioner for the
// multipliers, with the element operator applied by partial assembly.
//
// With -cg, the broken fluxes are instead eliminated, and the multiplier Schur
// complement is solved by CG, preconditioned by the graph preconditioner. This
// relies on fast diagonalization being the exact element inverse, which holds
// for the SPE10 problem (Cartesian mesh, elementwise constant coefficients).

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
#if defined(HYPRE_USING_GPU) && MFEM_HYPRE_VERSION >= 22100
   // Use hypre's SpMV kernels rather than the vendor library: cuSPARSE was much
   // slower for some of the AMG interpolation matrices.
   HYPRE_SetSpMVUseVendor(0);
#endif

   const char *device_config = "cpu";
   int order = 3;
   // Grid size. If ny, nz are negative, will be determined by nx
   int nx = 60;
   int ny = -1;
   int nz = -1;
   int ref_par = 0;
   real_t beta = 1.0;
   bool use_cg = false;
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&nx, "-nx", "--nx", "Number of cells in x dimension.");
   args.AddOption(&ny, "-ny", "--ny", "Number of cells in y dimension.");
   args.AddOption(&nz, "-nz", "--nz", "Number of cells in y dimension.");
   args.AddOption(&ref_par, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.AddOption(&beta, "-b", "--beta", "Div-div coefficient beta.");
   args.AddOption(&use_cg, "-cg", "--cg", "-no-cg", "--no-cg",
                  "Solve the multiplier Schur complement with CG (assuming "
                  "exact fast diagonalization), instead of the saddle-point "
                  "system with MINRES.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable ParaView.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = MakeSPE10Mesh(nx, ny, nz, ref_par); // handles negative ny, nz
   // Create the mesh nodes (used for the geometric factors) up front, so that
   // they are not included in the solver timing.
   mesh.EnsureNodes();
   const int dim = mesh.Dimension();

   RT_FECollection fec_rt(order-1, dim, BasisType::GaussLobatto,
                          BasisType::IntegratedGLL);
   ParFiniteElementSpace fes_rt(&mesh, &fec_rt);
   const HYPRE_BigInt ndofs_rt = fes_rt.GlobalTrueVSize();
   if (Mpi::Root()) { cout << "\nRT DOFs: " << ndofs_rt << endl; }

   Array<int> ess_tdofs;
   fes_rt.GetBoundaryTrueDofs(ess_tdofs);

   // Project the SPE10 coefficient to a piecewise constant coefficient grid
   // function (as in spe10.cpp).
   if (Mpi::Root()) cout << "Setting up coefficient... " << flush;
   SPE10Coefficient spe10_coeff;
   spe10_coeff.spe10.Invert();
   if (Mpi::Root()) cout << "Done.\nCoefficient space... " << flush;
   L2_FECollection fec_coeff(0, dim);
   ParFiniteElementSpace fes_coeff(&mesh, &fec_coeff, dim);
   ParGridFunction coeff_gf(&fes_coeff);
   if (Mpi::Root()) cout << "Done.\nProjecting coefficient... " << flush;
   coeff_gf.ProjectCoefficient(spe10_coeff);
   VectorGridFunctionCoefficient alpha_coeff(&coeff_gf);
   ConstantCoefficient beta_coeff(beta);
   if (Mpi::Root()) cout << "Done." << endl;

   tic_toc.Clear(); tic_toc.Start();
   if (Mpi::Root()) { cout << "\nHybridization solver... " << flush; }

   // Boundary values of the flux.
   ParGridFunction u_bc(&fes_rt);
   ProjectConstantFlux(Vector({1.0, 0.0, 0.0}), u_bc);

   BrokenSpace fes_hat(fes_rt);
   Array<int> offsets({0, fes_hat.GetNumBrokenDofs(),
                       fes_hat.GetNumBrokenDofs() + fes_hat.GetNumTraceDofs()});
   BlockVector rhs(offsets), solution(offsets);
   rhs.UseDevice(true);
   solution.UseDevice(true);
   rhs = 0.0; // zero right-hand side, except for the boundary conditions
   solution = 0.0;
   fes_hat.LocalProlongate(u_bc, solution.GetBlock(0));

   Array<int> ess_broken_dofs;
   fes_hat.GetBrokenEssentialDofs(ess_tdofs, ess_broken_dofs);

   BrokenHdivOperator A_hat(fes_rt, &alpha_coeff, &beta_coeff);
   // Eliminate the essential flux values u_e from the right-hand side:
   // g -= A u_e and g_e = u_e. (C has no entries in the essential columns.)
   {
      Vector &u = solution.GetBlock(0), &g = rhs.GetBlock(0);
      Vector u_e(u.Size()), A_u_e(u.Size()), values;
      u_e.UseDevice(true);
      A_u_e.UseDevice(true);
      u.GetSubVector(ess_broken_dofs, values);
      u_e = 0.0;
      u_e.SetSubVector(ess_broken_dofs, values);
      A_hat.Mult(u_e, A_u_e);
      g -= A_u_e;
      g.SetSubVector(ess_broken_dofs, values);
   }
   A_hat.SetEssentialDofs(fes_rt, ess_broken_dofs);

   GraphPreconditioner trace_inverse(fes_hat, ess_tdofs, &alpha_coeff,
                                     &beta_coeff);
   // Ct has the structure checked by VerifyConstraintStructure by construction;
   // the check is skipped here, since it runs on the host.
   HypreParMatrix &Ct = fes_hat.GetCt();
   const Array<int> &inactive_dofs = fes_hat.GetInactiveTraceDofs();
   FastDiagonalizationSolver element_inverse(fes_rt, ess_broken_dofs,
                                             &alpha_coeff, &beta_coeff);

   // Setup time, including the AMG setup. Synchronize the device and the
   // ranks, so that no setup work is attributed to the solve.
   MFEM_DEVICE_SYNC;
   MPI_Barrier(MPI_COMM_WORLD);
   const real_t setup_time = tic_toc.RealTime();

   int num_iterations;
   if (use_cg)
   {
      // Eliminate the broken fluxes using the exact element inverse A^{-1}:
      // solve (C A^{-1} C^T + I_i) lambda = C A^{-1} g, then recover
      // u = A^{-1} (g - C^T lambda). The multiplier right-hand side is zero,
      // and C has no rows for the inactive multipliers, so their right-hand
      // side entries are zero as well.
      Vector &g = rhs.GetBlock(0), &b = rhs.GetBlock(1);
      Vector &u = solution.GetBlock(0), &lambda = solution.GetBlock(1);
      Vector z(g.Size());
      z.UseDevice(true);
      element_inverse.Mult(g, z);
      Ct.MultTranspose(z, b);

      HybridSchurComplementOperator S(element_inverse, Ct, inactive_dofs);
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetPreconditioner(trace_inverse);
      cg.SetOperator(S);
      cg.SetRelTol(1e-12);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(500);
      cg.SetPrintLevel(IterativeSolver::PrintLevel().Warnings());
      cg.Mult(b, lambda);
      num_iterations = cg.GetNumIterations();

      z = g;
      Ct.Mult(-1.0, lambda, 1.0, z);
      element_inverse.Mult(z, u);
   }
   else
   {
      HybridSaddlePointOperator K(A_hat, Ct, inactive_dofs);
      BlockDiagonalPreconditioner preconditioner(offsets);
      preconditioner.SetDiagonalBlock(0, &element_inverse);
      preconditioner.SetDiagonalBlock(1, &trace_inverse);
      MINRESSolver minres(MPI_COMM_WORLD);
      minres.SetPreconditioner(preconditioner);
      minres.SetOperator(K);
      minres.SetRelTol(1e-12);
      minres.SetAbsTol(0.0);
      minres.SetMaxIter(500);
      minres.SetPrintLevel(IterativeSolver::PrintLevel().Warnings());
      minres.Mult(rhs, solution);
      num_iterations = minres.GetNumIterations();
   }

   MFEM_DEVICE_SYNC;
   MPI_Barrier(MPI_COMM_WORLD);
   const real_t elapsed = tic_toc.RealTime();

   if (Mpi::Root())
   {
      cout << "Done.\n"
           << "== Ranks: " << Mpi::WorldSize() << '\n'
           << "== Order: " << order << '\n'
           << "== nx: " << nx << '\n'
           << "== ny: " << ny << '\n'
           << "== nz: " << nz << '\n'
           << "== rp: " << ref_par << '\n'
           << "== beta: " << beta << '\n'
           << "== Solver: " << (use_cg ? "CG" : "MINRES") << '\n'
           << "== RT DOFs: " << ndofs_rt << '\n'
           << "== Iterations: " << num_iterations << '\n'
           << "== Setup: " << setup_time << '\n'
           << "== Solve: " << elapsed - setup_time << '\n'
           << "== Elapsed: " << elapsed << endl;
   }

   if (visualization)
   {
      // The broken fluxes are continuous at convergence; average them.
      Vector u_tdofs;
      fes_hat.Average(solution.GetBlock(0), u_tdofs);
      ParGridFunction x_rt(&fes_rt);
      x_rt.SetFromTrueDofs(u_tdofs);

      ParaViewDataCollection dc("SPE10_HB", &mesh);
      dc.SetPrefixPath("ParaView");
      dc.SetHighOrderOutput(true);
      dc.SetLevelsOfDetail(order);
      dc.RegisterField("u", &x_rt);
      dc.RegisterField("alpha", &coeff_gf);
      dc.Save();
   }

   return 0;
}
