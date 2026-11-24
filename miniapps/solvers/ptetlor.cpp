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
//   mpirun -np 2 ./ptetlor -g isaac -o 4 -e 2
//   mpirun -np 2 ./ptetlor -g duffy -o 4 -e 1 -t fit -ref
//   mpirun -np 2 ./ptetlor -g isaac -o 4 -e 2 -t entity -ref
//   mpirun -np 2 ./ptetlor -g electro -o 5 -e 0 -t interp
//   MFEM_GPU_AWARE_MPI=1 mpirun -np 2 ./ptetlor -d cuda -pa  # GPU-aware MPI

#include "tetlor.hpp"
#include <limits>

using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   std::string mesh_file = "../../data/tet-edge-star.mesh";
   std::string grid_type = "isaac";
   std::string transfer = "fit";
   bool reference = false;
   const char *device_config = "cpu";
   int p = 3;
   int enrichment = 2;
   int ref_levels = 0;
   bool pa = false;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device", "MFEM device configuration.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Conforming affine tetrahedral mesh.");
   args.AddOption(&p, "-o", "--order", "Polynomial degree.");
   args.AddOption(&enrichment, "-e", "--enrichment", "LOR order q = p + e.");
   args.AddOption(&grid_type, "-g", "--grid",
                  "LOR point distribution: isaac, duffy, or electro (q <= 10).");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of uniform coarse mesh refinements.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Use partial assembly for the high-order form.");
   args.AddOption(&transfer, "-t", "--transfer",
                  "Transfer method: fit, entity, or interp (square nodal grid).");
   args.AddOption(&reference, "-ref", "--reference", "-phys", "--physical",
                  "Geometry for fit and entity transfers.");
   args.ParseCheck();
   const auto method = tetlor::GetTransferMethod(transfer);

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }
   MFEM_VERIFY(!Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK) ||
               HypreUsingGPU(), "A GPU solve requires GPU-enabled hypre.");
   MFEM_VERIFY(p >= 1 && enrichment >= 0 && ref_levels >= 0 &&
               enrichment <= std::numeric_limits<int>::max() - p,
               "Invalid order or refinement.");
   const int q = p + enrichment;

   ParMesh mesh;
   {
      Mesh serial = Mesh::LoadFromFile(mesh_file, 1, 0);
      for (int r = 0; r < ref_levels; r++) { serial.UniformRefinement(); }
      MFEM_VERIFY(serial.GetNE() >= Mpi::WorldSize(),
                  "Use at most one rank per coarse element.");
      mesh = ParMesh(MPI_COMM_WORLD, serial);
   }

   H1Pos_FECollection bern(p);
   ParFiniteElementSpace high(&mesh, &bern);
   tetlor::BatchedLOR lor(mesh, q, grid_type);
   const auto nh = high.GlobalTrueVSize();
   const auto nl = static_cast<ParFiniteElementSpace&>
                   (lor.Space()).GlobalTrueVSize();
   if (Mpi::Root()) { tetlor::PrintProblemSize(p, q, nh, nl); }

   Array<int> ess_dofs, lor_ess_dofs;
   high.GetBoundaryTrueDofs(ess_dofs);
   lor.Space().GetBoundaryTrueDofs(lor_ess_dofs);

   ParBilinearForm a(&high);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.SetDiagonalPolicy(Operator::DIAG_ONE);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   ConstantCoefficient one(1.0);
   ParLinearForm b(&high);
   auto *load = new DomainLFIntegrator(one);
   if (pa)
   {
      // The constant load on affine elements needs degree-p exactness.
      load->SetIntegrationRule(StroudIntRules.Get(Geometry::TETRAHEDRON, p));
   }
   b.AddDomainIntegrator(load);
   b.UseFastAssembly(pa);
   b.Assemble();

   ParGridFunction x(&high);
   x = 0.0;
   OperatorPtr A;
   Vector X, B;
   a.FormLinearSystem(ess_dofs, x, b, A, X, B);

   tetlor::ElementTransfer R(high, lor, {method, reference});
   OperatorHandle Ahat(Operator::Hypre_ParCSR);
   lor.FormSystemMatrix(lor_ess_dofs, Ahat);
   HypreBoomerAMG Ahat_inv(*Ahat.As<HypreParMatrix>());
   if (HypreUsingGPU())
   {
      // Keep the coarse solve on the GPU too. The default Gaussian elimination
      // stages the coarse vectors on the host on each V-cycle.
      HYPRE_BoomerAMGSetCycleRelaxType(Ahat_inv, 18, 3); // l1-Jacobi
   }
   Ahat_inv.SetPrintLevel(1);
   tetlor::FictitiousSpaceSolver prec(R, Ahat_inv, ess_dofs);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetPreconditioner(prec);
   cg.SetOperator(*A);
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.Mult(B, X);

   return 0;
}
