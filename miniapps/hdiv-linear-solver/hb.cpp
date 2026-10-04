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

   return 0;
}
