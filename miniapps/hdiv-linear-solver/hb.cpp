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

// Solve the hybridized system with MINRES.
// Sample run: mpirun -np 2 hb -m ../../data/inline-quad.mesh -o 2

#include "mfem.hpp"
#include "hb.hpp"
#include <cmath>
#include <iostream>
#include <string>

using namespace mfem;
using namespace std;

void ExactSolution(const Vector &x, Vector &u)
{
   u = 0.0;
   u(0) = cos(M_PI*x(0))*sin(M_PI*x(1));
   u(1) = cos(M_PI*x(1))*sin(M_PI*x(0));
}

void RightHandSide(const Vector &x, Vector &f)
{
   ExactSolution(x, f);
   f *= 1.0 + 2.0*M_PI*M_PI;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int ser_ref = 0;
   int par_ref = 0;
   bool direct_solver = false;
   string trace_type = "graph";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&ser_ref, "-rs", "--refine-serial", "Serial mesh refinements.");
   args.AddOption(&par_ref, "-rp", "--refine-parallel",
                  "Parallel mesh refinements.");
   args.AddOption(&trace_type, "-t", "--trace",
                  "Trace solver: graph, hb, or lor.");
   args.AddOption(&direct_solver, "-d", "--direct", "-a",
                  "--amg", "Direct solver or AMG for HB system.");
   args.ParseCheck();
   MFEM_VERIFY(trace_type == "graph" || trace_type == "hb" || trace_type == "lor",
               "Trace solver must be graph, hb, or lor.");
   MFEM_VERIFY(!direct_solver || Mpi::WorldSize() == 1,
               "The direct solver requires one MPI rank.");

   ParMesh mesh = [&]()
   {
      Mesh serial_mesh(mesh_file);
      for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
      return ParMesh(MPI_COMM_WORLD, serial_mesh);
   }();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   const int dim = mesh.Dimension();
   RT_FECollection fec(order - 1, dim, BasisType::GaussLobatto,
                       BasisType::IntegratedGLL);
   ParFiniteElementSpace fes_rt(&mesh, &fec);
   const HYPRE_BigInt n_global_dofs = fes_rt.GlobalTrueVSize();
   if (Mpi::Root()) { cout << "Global RT DOFs: " << n_global_dofs << '\n'; }

   VectorFunctionCoefficient exact_coeff(dim, ExactSolution);
   VectorFunctionCoefficient rhs_coeff(dim, RightHandSide);

   Array<int> ess_tdofs;
   fes_rt.GetBoundaryTrueDofs(ess_tdofs);

   ParGridFunction x(&fes_rt);
   x.ProjectCoefficient(exact_coeff);

   ParLinearForm b(&fes_rt);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(rhs_coeff));
   b.Assemble();
   Vector f(fes_rt.GetTrueVSize());
   b.ParallelAssemble(f);

   ParBilinearForm a(&fes_rt);
   a.AddDomainIntegrator(new DivDivIntegrator);
   a.AddDomainIntegrator(new VectorFEMassIntegrator);

   BrokenSpace fes_hat(fes_rt);
   Array<int> offsets({0, fes_hat.GetNumBrokenDofs(),
                       fes_hat.GetNumBrokenDofs() + fes_hat.GetNumTraceDofs()});
   BlockVector rhs(offsets), solution(offsets);
   rhs = 0.0;
   solution = 0.0;

   fes_hat.AverageTranspose(f, rhs.GetBlock(0));
   fes_hat.LocalProlongate(x, solution.GetBlock(0));

   SparseMatrix A_hat(offsets[1]);
   DenseMatrix element_matrix;
   const int nd = fes_hat.GetNumDofsPerElem();
   Array<int> element_dofs(nd);
   for (int e = 0; e < fes_rt.GetNE(); ++e)
   {
      a.ComputeElementMatrix(e, element_matrix);
      for (int i = 0; i < nd; ++i) { element_dofs[i] = e*nd+i; }
      A_hat.AddSubMatrix(element_dofs, element_dofs, element_matrix);
   }
   A_hat.Finalize();

   auto stype = direct_solver ? DIRECT : AMG;
   unique_ptr<Solver> trace_inverse;
   if (trace_type == "graph")
   {
      trace_inverse.reset(new GraphPreconditioner(fes_hat, stype));
   }
   else
   {
      const auto amode = trace_type == "lor" ?
                         TraceMultiplierSolver::LOR : TraceMultiplierSolver::ACTUAL;
      trace_inverse.reset(new TraceMultiplierSolver(
                             a, ess_tdofs, amode, stype));
   }
   HypreParMatrix &Ct = [&]() -> HypreParMatrix &
   {
      if (trace_type == "graph") { return fes_hat.GetCt(); }
      return static_cast<TraceMultiplierSolver&>(*trace_inverse)
      .GetConstraintTranspose();
   }();
   TransposeOperator C(Ct);

   SparseMatrix inactive(fes_hat.GetNumTraceDofs());
   for (int td : fes_hat.GetInactiveTraceDofs()) { inactive.Add(td, td, 1.0); }
   inactive.Finalize();
   BlockOperator K(offsets);
   K.SetBlock(0, 0, &A_hat);
   K.SetBlock(0, 1, &Ct);
   K.SetBlock(1, 0, &C);
   K.SetBlock(1, 1, &inactive);

   Array<int> ess_broken_dofs;
   fes_hat.GetBrokenEssentialDofs(ess_tdofs, ess_broken_dofs);
   ConstrainedOperator constrained_K(&K, ess_broken_dofs);
   constrained_K.EliminateRHS(solution, rhs);
   BrokenElementSolver element_inverse(a, ess_broken_dofs);
   BlockDiagonalPreconditioner preconditioner(offsets);
   preconditioner.SetDiagonalBlock(0, &element_inverse);
   preconditioner.SetDiagonalBlock(1, trace_inverse.get());
   MINRESSolver minres(MPI_COMM_WORLD);
   minres.SetPreconditioner(preconditioner);
   minres.SetOperator(constrained_K);
   minres.SetRelTol(1e-10);
   minres.SetAbsTol(0.0);
   minres.SetMaxIter(1000);
   minres.SetPrintLevel(1);
   minres.Mult(rhs, solution);

   return 0;
}
