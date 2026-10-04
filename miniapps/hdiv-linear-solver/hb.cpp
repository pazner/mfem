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

// Solve the div-div plus mass problem with UMFPack on one MPI rank.
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

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.ParseCheck();

   ParMesh mesh = [&]()
   {
      Mesh serial_mesh(mesh_file);
      return ParMesh(MPI_COMM_WORLD, serial_mesh);
   }();
   const int dim = mesh.Dimension();
   RT_FECollection fec(order - 1, dim, BasisType::GaussLobatto,
                       BasisType::IntegratedGLL);
   ParFiniteElementSpace fes_rt(&mesh, &fec);
   const HYPRE_BigInt n_global_dofs = fes_rt.GlobalTrueVSize();
   if (Mpi::Root()) { cout << "Global RT DOFs: " << n_global_dofs << '\n'; }

   MFEM_VERIFY(Mpi::WorldSize() == 1, "");

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

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdofs, x, b, A, X, B);

   // On one rank, the diagonal block is the entire parallel matrix.
   SparseMatrix A_serial;
   A.As<HypreParMatrix>()->GetDiag(A_serial);
   UMFPackSolver solver;
   solver.SetOperator(A_serial);
   solver.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);
   cout << "L2 error: " << x.ComputeL2Error(exact_coeff) << '\n';

   BrokenSpace space(fes_rt);
   Array<int> offsets({0, space.restriction.Height(),
                       space.restriction.Height()+space.trace_fes.GetTrueVSize()});
   BlockVector rhs(offsets), solution(offsets);
   rhs = 0.0;
   solution = 0.0;

   space.AverageTranspose(f, rhs.GetBlock(0));
   space.restriction.Mult(x, solution.GetBlock(0));

   SparseMatrix A_hat(offsets[1]);
   DenseMatrix element_matrix;
   Array<int> element_dofs(space.nd);
   for (int e = 0; e < fes_rt.GetNE(); ++e)
   {
      a.ComputeElementMatrix(e, element_matrix);
      for (int i = 0; i < space.nd; ++i) { element_dofs[i] = e*space.nd+i; }
      A_hat.AddSubMatrix(element_dofs, element_dofs, element_matrix);
   }
   A_hat.Finalize();

   SparseMatrix Ct, inactive(space.trace_fes.GetTrueVSize());
   space.Ct->GetDiag(Ct);
   unique_ptr<SparseMatrix> C(Transpose(Ct));
   for (int td : space.inactive_trace) { inactive.Add(td, td, 1.0); }
   inactive.Finalize();
   BlockMatrix K(offsets);
   K.SetBlock(0, 0, &A_hat);
   K.SetBlock(0, 1, &Ct);
   K.SetBlock(1, 0, C.get());
   K.SetBlock(1, 1, &inactive);
   unique_ptr<SparseMatrix> K_serial(K.CreateMonolithic());

   Vector essential(fes_rt.GetTrueVSize()), broken_essential;
   essential = 0.0;
   for (int td : ess_tdofs) { essential(td) = 1.0; }
   space.Prolongate(essential, broken_essential);
   for (int i = 0; i < broken_essential.Size(); ++i)
   {
      if (broken_essential(i) != 0.0)
      {
         K_serial->EliminateRowCol(i, solution(i), rhs);
      }
   }
   solver.SetOperator(*K_serial);
   solver.Mult(rhs, solution);

   Vector recovered, difference(X), residual(rhs.Size()), jump(space.C->Height());
   space.Average(solution.GetBlock(0), recovered);
   difference -= recovered;
   K_serial->Mult(solution, residual);
   residual -= rhs;
   space.C->Mult(solution.GetBlock(0), jump);
   const real_t error = difference.Norml2()/max(real_t(1.0), X.Norml2());
   cout << "HB relative solution difference: " << error
        << "\nHB relative residual: "
        << residual.Norml2()/max(real_t(1.0), rhs.Norml2())
        << "\nHB continuity residual: " << jump.Norml2() << '\n';
   MFEM_VERIFY(error < 1e-10, "HB solution differs from the conforming solve.");

   return 0;
}
