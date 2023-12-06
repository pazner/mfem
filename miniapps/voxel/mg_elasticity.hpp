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

#ifndef MG_ELASTICITY_HPP
#define MG_ELASTICITY_HPP

#include "mfem.hpp"
#include "mg.hpp"

namespace mfem
{

class ImageElasticityMultigrid : public MultigridBase
{
public:
   std::vector<std::unique_ptr<ImageMesh>> meshes;
   std::vector<std::unique_ptr<FiniteElementSpace>> spaces;
   std::vector<std::unique_ptr<BilinearForm>> forms;
   std::vector<std::unique_ptr<Array<int>>> ess_dofs;
   std::vector<std::unique_ptr<ImageProlongation>> prolongations;
   int nlevels;
   ConstantCoefficient lambda{1.0};
   ConstantCoefficient mu{1.0};

   virtual const Operator* GetProlongationAtLevel(int level) const override
   {
      return prolongations[level].get();
   }

public:
   ImageElasticityMultigrid(const ImageMesh &&fine_mesh,
                            FiniteElementCollection &fec);

   FiniteElementSpace &GetFineSpace() { return *spaces.back(); }
   Operator &GetFineOperator() { return *operators.Last(); }
   BilinearForm &GetFineForm() { return *forms.back(); }
   void FormFineLinearSystem(
      Vector& x, Vector& b, OperatorHandle& A, Vector& X, Vector& B);
};

}

#endif
