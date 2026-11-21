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

// Tetrahedral fictitious-space preconditioning with an exact auxiliary solve.
//
//   make tetlor
//   ./tetlor -g isaac -p 4 -e 2
//   ./tetlor -g duffy -p 4 -e 1

#include "lor_mesh.hpp"
#include "tetlor.hpp"
#include <limits>

using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/tet-edge-star.mesh";
   std::string grid_type = "isaac";
   int p = 3, enrichment = 2, ref_levels = 0;

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
   MFEM_VERIFY(p >= 1 && enrichment >= 0 && ref_levels >= 0,
               "Invalid order or refinement.");
   MFEM_VERIFY(enrichment <= std::numeric_limits<int>::max() - p,
               "LOR order overflow.");
   const int q = p + enrichment;

   Mesh mesh = Mesh::LoadFromFile(mesh_file, 1, 0);
   for (int r = 0; r < ref_levels; r++) { mesh.UniformRefinement(); }
   Mesh mesh_lor = tetlor::MakeLORMesh(mesh, q, grid_type);

   H1Pos_FECollection bern(p);
   H1_FECollection linear(1);
   FiniteElementSpace high(&mesh, &bern), low(&mesh_lor, &linear);
   tetlor::PrintProblemSize(p, q, high.GetVSize(), low.GetVSize());

   Array<int> ess, lor_ess;
   high.GetBoundaryTrueDofs(ess);
   low.GetBoundaryTrueDofs(lor_ess);

   BilinearForm a(&high);
   a.SetDiagonalPolicy(Operator::DIAG_ONE);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   ConstantCoefficient one(1.0);
   LinearForm b(&high);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   GridFunction x(&high);
   x = 0.0;
   SparseMatrix A;
   Vector X, B;
   a.FormLinearSystem(ess, x, b, A, X, B);
   const real_t b_norm = B.Norml2();
   MFEM_VERIFY(b_norm > 0, "No free polynomial unknowns for the Dirichlet load.");

   BilinearForm w(&low);
   w.SetDiagonalPolicy(Operator::DIAG_ONE);
   w.AddDomainIntegrator(new DiffusionIntegrator);
   SparseMatrix R(high.GetVSize(), low.GetVSize());
   tetlor::AssembleTransfer(high, w, R);

   SparseMatrix W;
   w.FormSystemMatrix(lor_ess, W);
   UMFPackSolver inverse(W);
   tetlor::FictitiousSpaceSolver prec(R, inverse, ess);
   CGSolver cg;
   cg.SetPreconditioner(prec);
   cg.SetOperator(A);
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.Mult(B, X);

   return 0;
}
