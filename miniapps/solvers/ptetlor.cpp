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

// Parallel tetrahedral fictitious-space preconditioning with one BoomerAMG
// V-cycle on the LOR mesh.
//
//   make ptetlor
//   mpirun -np 2 ./ptetlor -g isaac -p 4 -e 2
//   mpirun -np 2 ./ptetlor -g duffy -p 4 -e 1

#include "lor_mesh.hpp"
#include "tetlor.hpp"
#include <algorithm>
#include <memory>
#include <limits>

using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   std::string mesh_file = "../../data/tet-edge-star.mesh";
   std::string grid_type = "isaac";
   int p = 3;
   int enrichment = 2;
   int ref_levels = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Conforming affine tetrahedral mesh.");
   args.AddOption(&p, "-p", "--order", "Polynomial degree.");
   args.AddOption(&enrichment, "-e", "--enrichment", "LOR order q = p + e.");
   args.AddOption(&grid_type, "-g", "--grid",
                  "LOR point distribution: isaac or duffy.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of uniform coarse mesh refinements.");
   args.ParseCheck();
   const int q = p + enrichment;

   ParMesh mesh, mesh_lor;
   {
      Mesh serial = Mesh::LoadFromFile(mesh_file, 1, 0);
      for (int r = 0; r < ref_levels; r++) { serial.UniformRefinement(); }
      Mesh refined = tetlor::MakeLORMesh(serial, q, grid_type);
      MFEM_VERIFY(serial.GetNE() >= Mpi::WorldSize(),
                  "Use at most one rank per coarse element.");
      // Keep every patch on its parent's rank. ParMesh preserves the relative
      // order of elements in each partition, hence also the q^3 child blocks.
      Array<int> parts;
      parts.MakeRef(serial.GeneratePartitioning(Mpi::WorldSize()),
                    serial.GetNE(), MemoryType::HOST, true);
      Array<int> children(refined.GetNE());
      const int n = refined.GetNE()/serial.GetNE();
      for (int e = 0; e < serial.GetNE(); e++)
      {
         std::fill_n(children.begin() + e*n, n, parts[e]);
      }
      mesh = ParMesh(MPI_COMM_WORLD, serial, parts);
      mesh_lor = ParMesh(MPI_COMM_WORLD, refined, children);
   }

   H1Pos_FECollection bern(p);
   H1_FECollection linear(1);
   ParFiniteElementSpace high(&mesh, &bern), low(&mesh_lor, &linear);
   const auto nh = high.GlobalTrueVSize(), nl = low.GlobalTrueVSize();
   if (Mpi::Root()) { tetlor::PrintProblemSize(p, q, nh, nl); }

   Array<int> ess, lor_ess;
   high.GetBoundaryTrueDofs(ess);
   low.GetBoundaryTrueDofs(lor_ess);

   ParBilinearForm a(&high);
   a.SetDiagonalPolicy(Operator::DIAG_ONE);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   ConstantCoefficient one(1.0);
   ParLinearForm b(&high);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   ParGridFunction x(&high);
   x = 0.0;
   HypreParMatrix A;
   Vector X, B;
   a.FormLinearSystem(ess, x, b, A, X, B);
   const real_t b_norm = ParNormlp(B, 2.0, MPI_COMM_WORLD);
   MFEM_VERIFY(b_norm > 0, "No free polynomial unknowns for the Dirichlet load.");

   ParBilinearForm w(&low);
   w.SetDiagonalPolicy(Operator::DIAG_ONE);
   w.AddDomainIntegrator(new DiffusionIntegrator);
   SparseMatrix local_R(high.GetVSize(), low.GetVSize());
   tetlor::AssembleTransfer(high, w, local_R);
   ParMixedBilinearForm transfer(&low, &high);
   std::unique_ptr<HypreParMatrix> R(transfer.ParallelAssemble(&local_R));

   HypreParMatrix W;
   w.FormSystemMatrix(lor_ess, W);
   HypreBoomerAMG inverse(W);
   inverse.SetPrintLevel(0);
   tetlor::FictitiousSpaceSolver prec(*R, inverse, ess);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetPreconditioner(prec);
   cg.SetOperator(A);
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.Mult(B, X);

   return 0;
}
