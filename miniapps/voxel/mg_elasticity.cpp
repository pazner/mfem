// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mg_elasticity.hpp"

namespace mfem
{

ImageElasticityMultigrid::ImageElasticityMultigrid(
   const ImageMesh &&fine_mesh, FiniteElementCollection &fec)
{
   meshes.emplace_back(new ImageMesh(fine_mesh));
   ImageMesh *current_mesh = meshes.back().get();
   const int dim = current_mesh->Dimension();

   while (current_mesh->GetImage().Width() >= 4 &&
          current_mesh->GetImage().Height() >= 4)
   {
      std::cout << "Current mesh: " << current_mesh->GetNE() << '\n';
      ImageMesh *new_mesh = new ImageMesh(current_mesh->Coarsen());
      std::cout << "New mesh:     " << new_mesh->GetNE() << '\n';
      meshes.emplace_back(new_mesh);
      current_mesh = new_mesh;
   }
   std::reverse(meshes.begin(), meshes.end());
   nlevels = meshes.size();

   for (int i = 0; i < nlevels; ++i)
   {
      spaces.emplace_back(new FiniteElementSpace(meshes[i].get(), &fec, dim));

      ess_dofs.emplace_back(new Array<int>);
      spaces[i]->GetBoundaryTrueDofs(*ess_dofs[i]);

      forms.emplace_back(new BilinearForm(spaces[i].get()));
      forms[i]->AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
      forms[i]->Assemble();

      OperatorHandle A;
      forms[i]->FormSystemMatrix(*ess_dofs[i], A);

      // DSmoother *smoother = new DSmoother(*A.As<SparseMatrix>(), 0, 0.8);
      GSSmoother *smoother = new GSSmoother(*A.As<SparseMatrix>());
      AddLevel(A.Ptr(), smoother, false, true);
   }

   for (int i = 0; i < nlevels - 1; ++i)
   {
      prolongations.emplace_back(
         new ImageProlongation(*spaces[i], *ess_dofs[i], *spaces[i+1], *ess_dofs[i+1]));
   }
}

void ImageElasticityMultigrid::FormFineLinearSystem(
   Vector &x, Vector &b, OperatorHandle &A, Vector& X, Vector& B)
{
   forms.back()->FormLinearSystem(*ess_dofs.back(), x, b, A, X, B);
}

}
