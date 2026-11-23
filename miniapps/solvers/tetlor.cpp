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

// Tetrahedral fictitious-space preconditioning with an exact auxiliary solve
// (UMFPACK on CPU, cuDSS on CUDA).
//
//   make tetlor
//   ./tetlor -g isaac -p 4 -e 2
//   ./tetlor -g duffy -p 4 -e 1 -t fit -ref
//   ./tetlor -g isaac -p 4 -e 2 -t entity -ref
//   ./tetlor -d cuda -pa  # requires cuDSS

#include "tetlor.hpp"
#include <limits>

using namespace mfem;

static std::unique_ptr<Solver> MakeAuxiliarySolver()
{
   if (Device::Allows(Backend::DEVICE_MASK))
   {
#ifdef MFEM_USE_CUDSS
      MFEM_VERIFY(Device::Allows(Backend::CUDA_MASK), "cuDSS requires CUDA.");
      return std::make_unique<CuDSSSolver>();
#else
      MFEM_ABORT("The serial device solve requires MFEM built with cuDSS.");
#endif
   }
#ifdef MFEM_USE_SUITESPARSE
   return std::make_unique<UMFPackSolver>();
#else
   MFEM_ABORT("The serial CPU solve requires MFEM built with SuiteSparse.");
#endif
}

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/tet-edge-star.mesh";
   std::string grid_type = "isaac";
   std::string transfer = "fit";
   bool reference = false;
   const char *device_config = "cpu";
   int p = 3, enrichment = 2, ref_levels = 0;
   bool pa = false;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device", "MFEM device configuration.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Conforming affine tetrahedral mesh.");
   args.AddOption(&p, "-p", "--order", "Polynomial degree.");
   args.AddOption(&enrichment, "-e", "--enrichment", "LOR order q = p + e.");
   args.AddOption(&grid_type, "-g", "--grid",
                  "LOR point distribution: isaac or duffy.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of uniform coarse mesh refinements.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Use partial assembly for the high-order form.");
   args.AddOption(&transfer, "-t", "--transfer",
                  "Transfer method: fit or entity.");
   args.AddOption(&reference, "-ref", "--reference", "-phys", "--physical",
                  "Build the transfer on the reference or physical geometry.");
   args.ParseCheck();
   MFEM_VERIFY(transfer == "fit" || transfer == "entity",
               "Expected transfer method fit or entity.");
   const auto method = transfer == "fit" ? tetlor::TransferMethod::Fit :
                       tetlor::TransferMethod::Entity;

   Device device(device_config);
   device.Print();
   auto Ahat_inv = MakeAuxiliarySolver();
   MFEM_VERIFY(p >= 1 && enrichment >= 0 && ref_levels >= 0,
               "Invalid order or refinement.");
   MFEM_VERIFY(enrichment <= std::numeric_limits<int>::max() - p,
               "LOR order overflow.");
   const int q = p + enrichment;

   Mesh mesh = Mesh::LoadFromFile(mesh_file, 1, 0);
   for (int r = 0; r < ref_levels; r++) { mesh.UniformRefinement(); }

   H1Pos_FECollection bern(p);
   FiniteElementSpace high(&mesh, &bern);
   tetlor::BatchedLOR lor(mesh, q, grid_type);
   tetlor::PrintProblemSize(p, q, high.GetVSize(), lor.Space().GetTrueVSize());

   Array<int> ess_dofs, lor_ess_dofs;
   high.GetBoundaryTrueDofs(ess_dofs);
   lor.Space().GetBoundaryTrueDofs(lor_ess_dofs);

   BilinearForm a(&high);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.SetDiagonalPolicy(Operator::DIAG_ONE);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   ConstantCoefficient one(1.0);
   LinearForm b(&high);
   auto *load = new DomainLFIntegrator(one);
   if (pa)
   {
      // The constant load on affine elements needs degree-p exactness.
      load->SetIntegrationRule(StroudIntRules.Get(Geometry::TETRAHEDRON, p));
   }
   b.AddDomainIntegrator(load);
   b.UseFastAssembly(pa);
   b.Assemble();

   GridFunction x(&high);
   x = 0.0;
   OperatorPtr A;
   Vector X, B;
   a.FormLinearSystem(ess_dofs, x, b, A, X, B);

   tetlor::ElementTransfer R(high, lor, {method, reference});
   OperatorHandle Ahat;
   lor.FormSystemMatrix(lor_ess_dofs, Ahat);
   Ahat_inv->SetOperator(*Ahat);
   tetlor::FictitiousSpaceSolver prec(R, *Ahat_inv, ess_dofs);

   CGSolver cg;
   cg.SetPreconditioner(prec);
   cg.SetOperator(*A);
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.Mult(B, X);

   return 0;
}
