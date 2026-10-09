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

#include "mfem.hpp"
#include <iostream>
#include <memory>

#include "discrete_divergence.hpp"
#include "hdiv_linear_solver.hpp"
#include "spe10.hpp"

using namespace std;
using namespace mfem;

// Solves the grad-div problem alpha u - grad(beta div u) = 0 on the SPE10
// domain, where alpha is the (diagonal, anisotropic) SPE10 coefficient and beta
// is constant, with the normal flux u.n = (1,0,0).n prescribed on the whole
// boundary. See also spe10_hb.cpp, which solves the same problem with the
// hybridization solver.

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *device_config = "cpu";
   int order = 3;
   // Grid size. If ny, nz are negative, will be determined by nx
   int nx = 60;
   int ny = -1;
   int nz = -1;
   int ref_par = 0;
   real_t beta = 1.0;
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
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable ParaView.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = MakeSPE10Mesh(nx, ny, nz, ref_par); // handles negative ny, nz
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");

   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLegendre;
   const int mt = FiniteElement::INTEGRAL;
   RT_FECollection fec_rt(order-1, dim, b1, b2);
   L2_FECollection fec_l2(order-1, dim, b2, mt);
   ParFiniteElementSpace fes_rt(&mesh, &fec_rt);
   ParFiniteElementSpace fes_l2(&mesh, &fec_l2);

   HYPRE_BigInt ndofs_rt = fes_rt.GlobalTrueVSize();
   HYPRE_BigInt ndofs_l2 = fes_l2.GlobalTrueVSize();

   if (Mpi::Root())
   {
      cout << "\nRT DOFs: " << ndofs_rt << "\nL2 DOFs: " << ndofs_l2 << endl;
   }

   Array<int> ess_rt_dofs;
   fes_rt.GetBoundaryTrueDofs(ess_rt_dofs);

   // Project the SPE10 coefficient to piecewise constant coefficient grid
   // function, which we then use as the coefficient for the problem.
   //
   // There are more efficient ways to do this: use a quadrature function and
   // set the values using element location (mesh geometric factors).
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

   // if (Mpi::Root()) cout << "Right-hand side... " << flush;
   // tic_toc.Clear();
   // tic_toc.Start();
   // ParLinearForm b_rt(&fes_rt);
   // b_rt.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_coeff));
   // b_rt.UseFastAssembly(true);
   // b_rt.Assemble();
   // if (Mpi::Root()) cout << "Done. " << tic_toc.RealTime() << endl;

   tic_toc.Clear(); tic_toc.Start();

   // if (Mpi::Root()) { cout << "\nSchur complement solver... " << flush; }
   if (Mpi::Root()) { cout << "\nSaddle point solver... " << flush; }
   HdivSaddlePointSolver saddle_point_solver(
      mesh, fes_rt, fes_l2, beta_coeff, alpha_coeff, ess_rt_dofs,
      HdivSaddlePointSolver::Mode::GRAD_DIV);

   ParGridFunction u_bc(&fes_rt);
   Vector g({1.0, 0.0, 0.0});
   VectorConstantCoefficient bc_coeff(g);
   u_bc.ProjectCoefficient(bc_coeff);
   Vector u_bc_tv(fes_rt.GetTrueVSize());
   u_bc.GetTrueDofs(u_bc_tv);
   saddle_point_solver.SetBC(u_bc_tv);

   const Array<int> &offsets = saddle_point_solver.GetOffsets();
   BlockVector X_block(offsets), B_block(offsets);

   // {
   //    HypreParMatrix &S = saddle_point_solver.GetApproxSchurComplement();
   //    HypreBoomerAMG &S_inv = saddle_point_solver.GetSchurComplementAMG();

   //    CGSolver cg(MPI_COMM_WORLD);
   //    cg.SetAbsTol(0.0);
   //    cg.SetRelTol(1e-12);
   //    cg.SetMaxIter(500);
   //    cg.SetPrintLevel(1);
   //    cg.SetOperator(S);
   //    cg.SetPreconditioner(S_inv);
   //    Vector X(fes_l2.GetTrueVSize()), B(fes_l2.GetTrueVSize());
   //    B.Randomize(1);
   //    X = 0.0;
   //    tic_toc.Clear();
   //    tic_toc.Start();
   //    cg.Mult(B, X);

   //    if (Mpi::Root())
   //    {
   //       cout << "Done.\nIterations: "
   //             << cg.GetNumIterations()
   //             << "\nElapsed: " << tic_toc.RealTime() << endl;
   //    }
   // }

   // B_block.GetBlock(0) = 0.0;
   // b_rt.ParallelAssemble(B_block.GetBlock(1));
   // B_block.GetBlock(1) *= -1.0;
   // B_block.SyncFromBlocks();

   B_block = 0.0; // zero RHS, except for BCs (eliminated in Mult)
   X_block = 0.0;

   // saddle_point_solver.GetMINRES().SetPrintLevel(1);
   saddle_point_solver.Mult(B_block, X_block);

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
           << "== RT DOFs: " << ndofs_rt << '\n'
           << "== L2 DOFs: " << ndofs_l2 << '\n'
           << "== Iterations: " << saddle_point_solver.GetNumIterations() << '\n'
           << "== Elapsed: " << tic_toc.RealTime() << endl;
   }

   if (visualization)
   {
      X_block.SyncToBlocks();

      ParGridFunction x_l2(&fes_l2), x_rt(&fes_rt);
      x_l2.SetFromTrueDofs(X_block.GetBlock(0));
      x_rt.SetFromTrueDofs(X_block.GetBlock(1));

      ParaViewDataCollection dc("SPE10", &mesh);
      dc.SetPrefixPath("ParaView");
      dc.SetHighOrderOutput(true);
      dc.SetLevelsOfDetail(order);
      dc.RegisterField("p", &x_l2);
      dc.RegisterField("u", &x_rt);
      dc.RegisterField("alpha", &coeff_gf);
      dc.Save();
   }

   return 0;
}
