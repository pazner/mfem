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

#ifndef VOXEL_INTEG_HPP
#define VOXEL_INTEG_HPP

#include "mfem.hpp"

namespace mfem
{

class VoxelIntegrator: public BilinearFormIntegrator
{
   std::unique_ptr<BilinearFormIntegrator> integ;
   DenseMatrix elmat;
   int ndof_per_el;
   int ne;
   mutable Vector z;
public:
   VoxelIntegrator(BilinearFormIntegrator *integ_);
   void AssemblePA(const FiniteElementSpace &fes) override;
   void AddMultPA(const Vector&, Vector&) const override;
   void AssembleDiagonalPA(Vector &diag) override;
};

}

#endif
