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

/// Quadrature rule of a QuadratureFunctionCoefficient, or nullptr otherwise.
/** Such coefficients can only be evaluated on their own rule. Assumes a mesh
    with a single element geometry. */
inline const IntegrationRule *GetCoefficientRule(Coefficient *coeff)
{
   auto *qf_coeff = dynamic_cast<QuadratureFunctionCoefficient*>(coeff);
   if (!qf_coeff) { return nullptr; }
   auto *qs = dynamic_cast<const QuadratureSpace*>(
                 qf_coeff->GetQuadFunction().GetSpace());
   MFEM_VERIFY(qs, "Expected an element quadrature space.");
   return &qs->GetIntRule(0);
}

/// Approximate element inverse of alpha (u,v) + beta (div u,div v) for RT cells.
/** Uses lexicographic, element-contiguous broken coordinates (the layout of
    MFEM's lexicographic element restriction, including RT signs).
    Approximates alpha J^T J/det J by its elementwise averaged diagonal and
    beta/det J by its elementwise average, using reference quadrature weights.
    Exact for affine orthogonal maps and elementwise constant coefficients; an
    SPD preconditioner otherwise. The coefficients alpha (mass) and beta (div)
    default to one when null. Requires LAPACK, uniform order,
    GaussLobatto/IntegratedGLL bases, positive Jacobians, alpha > 0 and
    beta >= 0, and order + 1 at most the H(div) limit HDIV_MAX_D1D of the
    configured device. Essential DOFs must be unions of complete element faces,
    in broken numbering; their equations are identity. Setup is on the host. Mult runs on the device with one thread
    block per element, holding the element vector in shared memory. Reconstruct
    the solver after changing the mesh or coefficients. */
class FastDiagonalizationSolver : public Solver
{
   int dim, p, nd, ne;
   /// Constraint mask of each element and direction, at index e*dim+a. Bits
   /// select constrained left/right endpoints.
   Array<int> masks;
   /// 1D bases for each of the 4 masks, each laid out as the column-major
   /// closed (p+1)x(p+1) and open pxp transforms followed by the p singular
   /// values. Constrained closed spaces are padded with zeros to retain the RT
   /// layout.
   Vector basis;
   DenseMatrix mass_inverse;
   Vector div_scale;

public:
   FastDiagonalizationSolver(FiniteElementSpace &fes,
                             const Array<int> &ess_dofs = Array<int>(),
                             Coefficient *alpha = nullptr,
                             Coefficient *beta = nullptr);

   void Mult(const Vector &x, Vector &y) const override;
   void SetOperator(const Operator &) override
   { MFEM_ABORT("FastDiagonalizationSolver must be reconstructed."); }
};

} // namespace mfem

#endif
