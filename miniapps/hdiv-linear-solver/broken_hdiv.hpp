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

#ifndef MFEM_HDIV_BROKEN_HDIV_HPP
#define MFEM_HDIV_BROKEN_HDIV_HPP

#include "mfem.hpp"

namespace mfem
{

/// Element-wise alpha (u,v) + beta (div u,div v) on the broken RT space.
/** Acts on lexicographic, element-contiguous broken coordinates (the layout of
    MFEM's lexicographic element restriction, as for partial assembly). The
    mass and div-div terms are applied by a single fused shared memory kernel.
    In 3D, each element is processed by a 1D thread block that flattens every
    stage over all fields; the sum factorization contractions assign one
    "pencil" (a line along the contracted direction, held in registers) to
    each thread.

    With GaussLobatto/IntegratedGLL bases, the reference divergence of the
    closed basis is exactly the open (integrated) basis composed with the
    signed difference of neighboring closed coefficients. The divergence is
    therefore formed from the element coefficients by differences, and the
    div-div term reduces to a scalar mass on the open tensor basis, instead of
    derivative contractions for every vector component.

    The quadrature rule is that of MassIntegrator (as used by the PA
    VectorFEMassIntegrator and DivDivIntegrator), or the rule of
    QuadratureFunctionCoefficient arguments. The coefficients alpha (mass) and
    beta (div) default to one when null. Requires uniform-order quadrilateral or
    hexahedral elements with positive Jacobians, within the H(div) limits
    HDIV_MAX_D1D and HDIV_MAX_Q1D of the configured device. Reconstruct the
    operator after changing the mesh or coefficients. */
class BrokenHdivOperator : public Operator
{
   int dim, ne, d1d, q1d;
   const Array<real_t> *Bo, *Bc; // open and closed 1D bases at quadrature
   /// Per quadrature point: the symmetric entries of alpha w J^T J/det J
   /// (row-major upper triangle), followed by beta w/det J.
   Vector qdata;

public:
   BrokenHdivOperator(const FiniteElementSpace &fes,
                      Coefficient *alpha = nullptr,
                      Coefficient *beta = nullptr);

   void Mult(const Vector &x, Vector &y) const override;
};

} // namespace mfem

#endif
