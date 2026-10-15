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

// Hybridized H(div) diffusion with a graph trace preconditioner.
// Sample runs (from this directory):
//   mpirun -np 2 ./hb_simple -m ../../../data/inline-quad.mesh -o 2 -rs 2
//   mpirun -np 2 ./hb_simple -m ../../../data/inline-hex.mesh -o 3 -d cuda

#include "hb.hpp"
#include "fast_diag.hpp"
#include "broken_hdiv.hpp"
#include <cmath>
#include <iostream>
#include <string>

using namespace mfem;
using namespace std;

static void ExactSolution(const Vector &x, Vector &u)
{
   u = 0.0;
   u(0) = cos(M_PI*x(0))*sin(M_PI*x(1));
   u(1) = cos(M_PI*x(1))*sin(M_PI*x(0));
}

static void RightHandSide(const Vector &x, Vector &f)
{
   ExactSolution(x, f);
   f *= 1.0 + 2.0*M_PI*M_PI;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

#if defined(HYPRE_USING_GPU) && MFEM_HYPRE_VERSION >= 22100
   // Native hypre kernels are faster for the AMG interpolation matrices.
   HYPRE_SetSpMVUseVendor(0);
#endif

   string mesh_file = "../../../data/inline-quad.mesh";
   string device_config = "cpu";
   int order = 1;
   int ser_ref = 0;
   int par_ref = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order", "Polynomial degree (at least one).");
   args.AddOption(&ser_ref, "-rs", "--refine-serial", "Serial mesh refinements.");
   args.AddOption(&par_ref, "-rp", "--refine-parallel",
                  "Parallel mesh refinements.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = [&]()
   {
      Mesh serial_mesh(mesh_file);
      for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
      return ParMesh(MPI_COMM_WORLD, serial_mesh);
   }();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   const int dim = mesh.Dimension();

   RT_FECollection fec(order-1, dim, BasisType::GaussLobatto,
                       BasisType::IntegratedGLL);
   ParFiniteElementSpace fes(&mesh, &fec);
   const HYPRE_BigInt global_dofs = fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Global RT DOFs: " << global_dofs << '\n';
   }

   const string suffix = "CrookedPipeMesh_1.mesh";
   const bool pipe = mesh_file.size() >= suffix.size() &&
                     mesh_file.compare(mesh_file.size() - suffix.size(),
                                       suffix.size(), suffix) == 0;

   // alpha (u,v) + beta (div u,div v). Attribute 1 is the pipe.
   Vector alpha_values(mesh.attributes.Max()), beta_values(mesh.attributes.Max());
   alpha_values = 1.0;
   beta_values = 1.0;
   if (pipe)
   {
      alpha_values = 0.2;
      alpha_values(0) = 2000.0;
      beta_values = 1.64098370E+00;
      beta_values(0) = 1.88046595E-03;
   }
   PWConstCoefficient alpha(alpha_values), beta(beta_values);
   VectorFunctionCoefficient exact(dim, ExactSolution);
   VectorFunctionCoefficient manufactured_rhs(dim, RightHandSide);
   Vector ones(dim);
   ones = 1.0;
   VectorConstantCoefficient pipe_rhs(ones);
   VectorCoefficient &source = pipe ? static_cast<VectorCoefficient&>(pipe_rhs)
                               : manufactured_rhs;

   // The manufactured problem prescribes the exact normal flux on the boundary.
   Array<int> ess_tdofs;
   ParGridFunction x(&fes);
   x = 0.0;
   if (!pipe)
   {
      fes.GetBoundaryTrueDofs(ess_tdofs);
      x.ProjectCoefficient(exact);
   }
   ParLinearForm b(&fes);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(source));
   b.UseFastAssembly(true);
   b.Assemble();
   Vector f(fes.GetTrueVSize());
   b.ParallelAssemble(f);

   StopWatch timer;
   MFEM_DEVICE_SYNC;
   MPI_Barrier(MPI_COMM_WORLD);
   timer.Start();
   BrokenSpace broken(fes);
   Array<int> offsets({0, broken.GetNumBrokenDofs(),
                       broken.GetNumBrokenDofs() + broken.GetNumTraceDofs()});
   BlockVector rhs(offsets), solution(offsets);
   rhs.UseDevice(true);
   solution.UseDevice(true);
   rhs = 0.0;
   solution = 0.0;
   rhs.SyncToBlocks();
   solution.SyncToBlocks();
   broken.AverageTranspose(f, rhs.GetBlock(0));
   broken.LocalProlongate(x, solution.GetBlock(0));

   Array<int> ess_broken_dofs;
   broken.GetBrokenEssentialDofs(ess_tdofs, ess_broken_dofs);
   BrokenHdivOperator A(fes, alpha, beta);
   // Eliminate boundary values once; enforce identity equations in the PA kernel.
   {
      ConstrainedOperator constrained(&A, ess_broken_dofs);
      constrained.EliminateRHS(solution.GetBlock(0), rhs.GetBlock(0));
   }
   A.SetEssentialDofs(fes, ess_broken_dofs);

   FastDiagonalizationSolver element_inverse(fes, ess_broken_dofs, alpha, beta);
   GraphPreconditioner trace_inverse(broken, ess_tdofs, alpha, beta);
   // C omits boundary faces; give their unused multipliers identity equations.
   EssentialDofIdentity inactive_identity(broken.GetNumTraceDofs(),
                                          broken.GetInactiveTraceDofs());

   BlockOperator K(offsets);
   K.SetDiagonalBlock(0, &A);
   K.SetBlock(0, 1, &broken.GetCt());
   K.SetBlock(1, 0, &broken.GetC());
   K.SetDiagonalBlock(1, &inactive_identity);
   BlockDiagonalPreconditioner preconditioner(offsets);
   preconditioner.SetDiagonalBlock(0, &element_inverse);
   preconditioner.SetDiagonalBlock(1, &trace_inverse);

   MINRESSolver minres(MPI_COMM_WORLD);
   minres.SetPreconditioner(preconditioner);
   minres.SetOperator(K);
   minres.SetRelTol(1e-12);
   minres.SetAbsTol(0.0);
   minres.SetMaxIter(1000);
   minres.SetPrintLevel(3);
   MFEM_DEVICE_SYNC;
   MPI_Barrier(MPI_COMM_WORLD);
   const real_t setup_time = timer.RealTime();
   rhs.SyncFromBlocks();
   solution.SyncFromBlocks();
   minres.Mult(rhs, solution);
   solution.SyncToBlocks();
   MFEM_DEVICE_SYNC;
   MPI_Barrier(MPI_COMM_WORLD);
   const real_t elapsed = timer.RealTime();
   if (Mpi::Root())
   {
      cout << "Setup: " << setup_time << '\n'
           << "Solve: " << elapsed-setup_time << '\n'
           << "Elapsed: " << elapsed << '\n';
   }

   // Recover the conforming solution by averaging the element copies.
   Vector true_solution;
   broken.Average(solution.GetBlock(0), true_solution);
   x.SetFromTrueDofs(true_solution);
   if (!pipe)
   {
      const real_t error = x.ComputeL2Error(exact);
      if (Mpi::Root()) { cout << "L2 error: " << error << '\n'; }
   }
   return minres.GetConverged() ? 0 : 1;
}
