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

#ifndef MFEM_HDIV_FAST_DIAG_HPP
#define MFEM_HDIV_FAST_DIAG_HPP

#include "mfem.hpp"

namespace mfem
{

/// Approximate element inverse of alpha (u,v) + beta (div u,div v) for RT cells.
/** Uses native, element-contiguous broken coordinates, including RT signs.
    Approximates alpha J^T J/det J by its elementwise averaged diagonal and
    beta/det J by its elementwise average, using reference quadrature weights.
    Exact for affine orthogonal maps; an SPD preconditioner otherwise.
    Requires LAPACK, uniform order, GaussLobatto/IntegratedGLL bases, positive
    Jacobians, constant alpha > 0 and beta >= 0. Essential DOFs must be
    unions of complete element faces, in broken numbering; their equations
    are identity. Setup is on the host, and Mult uses host tensor contractions.
    Reconstruct the solver after changing the mesh or coefficients. */
class FastDiagonalizationSolver : public Solver
{
   struct Basis
   {
      // Pad constrained closed spaces with zeros to retain the RT layout.
      DenseMatrix closed, open;
      Vector sigma;
   } basis[4]; // Bits select constrained left/right endpoints.

   int dim, p, nd, ne;
   Array<int> dof_map;
   Array2D<int> masks;
   DenseMatrix mass_inverse;
   Vector div_scale;
   mutable Vector work, scratch;

   void Transform(int e, bool transpose) const;

public:
   FastDiagonalizationSolver(FiniteElementSpace &fes,
                             const Array<int> &ess_dofs = Array<int>(),
                             real_t alpha = 1.0, real_t beta = 1.0);

   void Mult(const Vector &x, Vector &y) const override;
   void SetOperator(const Operator &) override
   { MFEM_ABORT("FastDiagonalizationSolver must be reconstructed."); }
};

} // namespace mfem

#endif
