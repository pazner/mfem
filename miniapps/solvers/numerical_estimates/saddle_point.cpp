// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//                     ---------------------------------
//                     H(div) saddle-point system solver
//                     ---------------------------------
//

#include "mfem.hpp"
#include "linalg/dtensor.hpp"
#include <fstream>
#include <iostream>
#include <memory>

#include "discrete_divergence.hpp"

using namespace std;
using namespace mfem;

Mesh LoadMesh(const char *mesh_file, int ser_ref = 0)
{
   Mesh mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { mesh.UniformRefinement(); }
   return mesh;
}

Mesh LoadMeshLOR(const char *mesh_file, int ser_ref, int order)
{
   Mesh mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { mesh.UniformRefinement(); }
   return Mesh::MakeRefined(mesh, order, BasisType::GaussLobatto);
}

void SaveMatrix(const SparseMatrix &A, const char *fname)
{
   std::ofstream f(fname);
   A.PrintMatlab(f);
}

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/ref-square.mesh";
   int ser_ref = 1;
   int order = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.ParseCheck();

   // Mesh mesh = LoadMesh(mesh_file, ser_ref);
   Mesh mesh = LoadMeshLOR(mesh_file, ser_ref, order);

   order = 1;

   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");

   // int b1 = BasisType::GaussLobatto, b2 = BasisType::IntegratedGLL;
   int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLegendre;
   int mt = FiniteElement::INTEGRAL;

   RT_FECollection fec_rt(order-1, dim, b1, b2);
   FiniteElementSpace fes_rt(&mesh, &fec_rt);

   L2_FECollection fec_l2(order-1, dim, b2, mt);
   FiniteElementSpace fes_l2(&mesh, &fec_l2);

   BilinearForm norm_rt(&fes_rt);
   norm_rt.AddDomainIntegrator(new VectorFEMassIntegrator);
   norm_rt.Assemble();
   norm_rt.Finalize();
   SaveMatrix(norm_rt.SpMat(), "M.txt");

   MixedBilinearForm div(&fes_rt, &fes_l2);
   div.AddDomainIntegrator(new MixedScalarDivergenceIntegrator);
   div.Assemble();
   div.Finalize();
   SaveMatrix(div.SpMat(), "B.txt");

   BilinearForm norm_l2(&fes_l2);
   norm_l2.AddDomainIntegrator(new MassIntegrator);
   norm_l2.Assemble();
   norm_l2.Finalize();
   SaveMatrix(norm_l2.SpMat(), "W.txt");

   return 0;
}
