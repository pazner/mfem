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

#ifndef MFEM_MINIAPPS_TETLOR_ENTITY_HPP
#define MFEM_MINIAPPS_TETLOR_ENTITY_HPP

#include "tetlor.hpp"

namespace tetlor
{

std::vector<BarycentricLabel> BernsteinLabels(const FiniteElement &bern);

std::unique_ptr<Operator> MakeEntityTransfer(FiniteElementSpace &high,
                                             BatchedLOR &lor);

namespace entity
{
// Free DOFs on the closed entity, interior first. Each of its two lists is
// ordered by true DOF number so incident cells and ranks assemble the same form.
struct EntityDofs
{
   Array<int> all;
   int interior = 0;

   void Init(FiniteElementSpace &fes, int dim, int id,
             const Array<int> &essential);
};

Array<int> BoundaryMarker(const FiniteElementSpace &fes);
std::unique_ptr<Operator> MakeDeviceTransfer(FiniteElementSpace &high,
                                             BatchedLOR &lor);

void ReferenceMatrix(BatchedLOR &lor, const FiniteElement &bern,
                     DenseMatrix &R);

// Polar map for open-entity residuals. Align the residuals of the constant
// function for the prescribed source and target trace lifts.
DenseMatrix TracePolar(const DenseMatrix &low, const DenseMatrix &high,
                       const DenseMatrix &sampling, DenseMatrix &low_lift,
                       DenseMatrix &high_lift);

// Schur complement on keep, minimizing over eliminate. Other DOFs are fixed.
DenseMatrix Condense(const DenseMatrix &K, const Array<int> &keep,
                     const Array<int> &eliminate);

// Interior DOFs precede boundary DOFs in each form. The returned matrices map
// LOR entity data and the previously transferred HO boundary to the HO interior.
void PolarBlocks(const DenseMatrix &low, const DenseMatrix &high,
                 const DenseMatrix &sampling, DenseMatrix &from_low,
                 DenseMatrix &from_boundary);
}
}
#endif
