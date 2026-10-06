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

// Compare the hybridized MINRES solve with UMFPack on one MPI rank.
// Sample run: mpirun -np 1 hb -m ../../data/inline-quad.mesh -o 2

#include "mfem.hpp"
#include "hb.hpp"
#include <cmath>
#include <iostream>

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
   int ref = 0;
   bool direct_solver = false;
   bool run_comparison = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&ref, "-r", "--refine", "Mesh refinements.");
   args.AddOption(&direct_solver, "-d", "--direct", "-a",
                  "--amg", "Direct solver or AMG for HB system.");
   args.AddOption(&run_comparison, "-c", "--compare", "-no-c",
                  "--no-comparison", "Compare with standard RT system.");
   args.ParseCheck();

   ParMesh mesh = [&]()
   {
      Mesh serial_mesh(mesh_file);
      for (int i = 0; i < ref; ++i) { serial_mesh.UniformRefinement(); }
      return ParMesh(MPI_COMM_WORLD, serial_mesh);
   }();
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
   Vector f;
   b.ParallelAssemble(f); // Save the load before essential elimination.

   ParBilinearForm a(&fes_rt);
   a.AddDomainIntegrator(new DivDivIntegrator);
   a.AddDomainIntegrator(new VectorFEMassIntegrator);
   a.Assemble();

   Vector B, X;

   if (run_comparison)
   {
      MFEM_VERIFY(Mpi::WorldSize() == 1, "");

      OperatorPtr A;
      a.FormLinearSystem(ess_tdofs, x, b, A, X, B);

      // On one rank, the diagonal block is the entire parallel matrix.
      SparseMatrix A_serial;
      A.As<HypreParMatrix>()->GetDiag(A_serial);
      UMFPackSolver solver;
      solver.SetOperator(A_serial);
      solver.Mult(B, X);

      a.RecoverFEMSolution(X, b, x);
      cout << "L2 error: " << x.ComputeL2Error(exact_coeff) << '\n';
   }

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

   HypreParMatrix &C_par = fes_hat.GetC();
   HypreParMatrix &Ct_par = fes_hat.GetCt();
   SparseMatrix C, Ct;
   C_par.GetDiag(C);
   Ct_par.GetDiag(Ct);

   SparseMatrix inactive(fes_hat.GetNumTraceDofs());
   for (int td : fes_hat.GetInactiveTraceDofs()) { inactive.Add(td, td, 1.0); }
   inactive.Finalize();
   BlockMatrix K(offsets);
   K.SetBlock(0, 0, &A_hat);
   K.SetBlock(0, 1, &Ct);
   K.SetBlock(1, 0, &C);
   K.SetBlock(1, 1, &inactive);
   unique_ptr<SparseMatrix> K_mat(K.CreateMonolithic());

   Array<int> ess_broken_dofs;
   fes_hat.GetBrokenEssentialDofs(ess_tdofs, ess_broken_dofs);
   for (int i : ess_broken_dofs)
   {
      K_mat->EliminateRowCol(i, solution(i), rhs);
   }
   BrokenElementSolver element_inverse(a, ess_broken_dofs);
   auto stype = direct_solver ? GraphPreconditioner::DIRECT :
                GraphPreconditioner::AMG;
   GraphPreconditioner trace_inverse(fes_hat, stype);
   BlockDiagonalPreconditioner preconditioner(offsets);
   preconditioner.SetDiagonalBlock(0, &element_inverse);
   preconditioner.SetDiagonalBlock(1, &trace_inverse);
   MINRESSolver minres(MPI_COMM_WORLD);
   minres.SetPreconditioner(preconditioner);
   minres.SetOperator(*K_mat);
   minres.SetRelTol(1e-10);
   minres.SetAbsTol(0.0);
   minres.SetMaxIter(1000);
   minres.SetPrintLevel(1);
   minres.Mult(rhs, solution);

   if (run_comparison)
   {
      Vector recovered;
      fes_hat.Average(solution.GetBlock(0), recovered);

      Vector difference = X;
      difference -= recovered;
      const real_t error = difference.Norml2()/max(real_t(1.0), X.Norml2());

      Vector residual(K_mat->Height());
      K_mat->Mult(solution, residual);
      residual -= rhs;

      Vector jump(fes_hat.GetNumTraceDofs());
      C.Mult(solution.GetBlock(0), jump);

      cout << "HB MINRES iterations: " << minres.GetNumIterations()
           << "\nHB relative solution difference: " << error
           << "\nHB relative residual: "
           << residual.Norml2()/max(real_t(1.0), rhs.Norml2())
           << "\nHB continuity residual: " << jump.Norml2() << '\n';
      // MFEM_VERIFY(error < 1e-10, "HB solution differs from the conforming solve.");
   }

   return 0;
}
