// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef PATCH_MULTIGRID_HPP
#define PATCH_MULTIGRID_HPP

#include "mfem.hpp"

namespace mfem
{

struct PatchSmoother : Solver
{
   std::vector<std::unique_ptr<Operator>> patches_inv;
   std::vector<Array<int>> patch_dofs;
   const Array<int> &ess_dofs; ///< Essential DOFs.
   const real_t alpha; ///< Damping parameter.
   int max_valence;
   mutable Vector z1, z2; ///< Temporary workspace vectors.

   PatchSmoother(const FiniteElementSpace &fes,
                 const SparseMatrix &A,
                 const Array<int> &ess_dofs_,
                 real_t alpha_=1.0);
   void SetOperator(const Operator &op);
   void Mult(const Vector &b, Vector &x) const;
   void MultTranspose(const Vector &b, Vector &x) const;
};

class PatchMultigrid : public GeometricMultigrid
{
   ConstantCoefficient lambda;

   void FormLevel(FiniteElementSpace &fespace, int level);
public:
   PatchMultigrid(FiniteElementSpaceHierarchy& fespaces,
                  Array<int>& ess_bdr,
                  real_t lambda_=1.0);
};

} // namespace mfem

#endif
