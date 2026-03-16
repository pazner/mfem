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

#ifndef MFEM_FE_H1_BUBBLE
#define MFEM_FE_H1_BUBBLE

#include "fe_base.hpp"
#include "fe_h1.hpp"

namespace mfem
{

/// Arbitrary order H1 Duffy triangular element
class H1Duffy_TriangleElement : public NodalFiniteElement
{
private:
   H1_QuadrilateralElement quad_fe;

public:
   H1Duffy_TriangleElement(int p, int btype = BasisType::GaussLobatto);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

} // namespace mfem

#endif
