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

// Placeholder for a parallel H(div) hybridization solver.
// Sample run: mpirun -np 4 hb -m ../../data/inline-quad.mesh -o 2

#include "mfem.hpp"
#include "hb.hpp"
#include <iostream>

using namespace mfem;
using namespace std;

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

   BrokenSpace fes_hat(fes_rt);

   // Array<int> ess_vdofs, ess_tdofs;
   // if (mesh.bdr_attributes.Size())
   // {
   //    Array<int> ess_bdr(mesh.bdr_attributes.Max());
   //    ess_bdr = 1;
   //    fes_rt.GetEssentialVDofs(ess_bdr, ess_vdofs);
   //    fes_rt.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
   // }

   return 0;
}
