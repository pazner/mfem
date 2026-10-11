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

#ifndef MFEM_HDIV_SIMPLE_FAST_DIAG_HPP
#define MFEM_HDIV_SIMPLE_FAST_DIAG_HPP

#include "mfem.hpp"

namespace mfem
{

/// Constraint masks of the faces of each element of the RT space fes, in the
/// lexicographic broken coordinates: masks[e*dim+c] has bits 0 and 1 set when
/// the faces at the left and right ends of direction c of element e are
/// constrained. The essential DOFs (broken indices) must be unions of complete
/// element faces. The space must have uniform-order tensor-product elements.
void GetConstrainedFaceMasks(const FiniteElementSpace &fes,
                             const Array<int> &ess_dofs, Array<int> &masks);

/// Approximate element inverse of alpha (u,v) + beta (div u,div v) for RT cells.
/** Uses lexicographic, element-contiguous broken coordinates (the layout of
    MFEM's lexicographic element restriction, including RT signs).
    Approximates J^T alpha J/det J by its elementwise averaged diagonal and
    beta/det J by its elementwise average, using reference quadrature weights.
    Exact for affine orthogonal maps and elementwise constant coefficients; an
    SPD preconditioner otherwise. The mass coefficient alpha and div-div
    coefficient beta are scalar. Requires LAPACK, uniform order,
    GaussLobatto/IntegratedGLL bases, positive Jacobians, alpha > 0 and
    beta > 0, and order + 1 at most the H(div) limit HDIV_MAX_D1D of the
    configured device. Essential DOFs must be unions of complete element faces,
    in broken numbering; their equations are identity. Setup runs on the device,
    except for the 1D bases. Mult runs on the device with one thread block per
    element, holding the element vector in shared memory. Reconstruct
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
                             const Array<int> &ess_dofs,
                             Coefficient &alpha, Coefficient &beta);

   void Mult(const Vector &x, Vector &y) const override;
   void SetOperator(const Operator &) override;
};

} // namespace mfem

#endif
