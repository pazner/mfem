// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
#include "patch_multigrid.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../../data/star.mesh";
   const char *device_config = "cpu";
   int ref = 0;
   int order = 2;
   real_t lambda = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref, "-r", "--refine", "Number of mesh refinements.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&lambda, "-l", "--lambda", "Grad-div coefficient.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh mesh(mesh_file);
   const int dim = mesh.Dimension();

   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLegendre;
   RT_FECollection fec(order-1, dim, b1, b2);

   // H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   FiniteElementSpaceHierarchy fes_hierarchy(&mesh, &fes, false, false);
   for (int i = 0; i < ref; ++i) { fes_hierarchy.AddUniformlyRefinedLevel(); }

   Array<int> ess_bdr;
   if (mesh.bdr_attributes.Size() > 0)
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
   }

   PatchMultigrid mg(fes_hierarchy, ess_bdr, lambda);

   Operator &A = *mg.GetOperatorAtFinestLevel();
   Vector B(A.Height());
   Vector X(A.Height());

   B.Randomize(1);
   X = 0.0;

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(A);
   cg.SetPreconditioner(mg);
   cg.Mult(B, X);

   return 0;
}
