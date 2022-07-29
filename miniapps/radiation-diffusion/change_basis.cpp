// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "change_basis.hpp"
#include "fem/qinterp/dispatch.hpp"

#include "general/nvtx.hpp"

namespace mfem
{

ChangeOfBasis_L2::ChangeOfBasis_L2(FiniteElementSpace &fes1,
                                   FiniteElementSpace &fes2)
   : Operator(fes1.GetTrueVSize()),
     ne(fes1.GetNE())
{
   const FiniteElement *fe = fes1.GetFE(0);
   const auto mode = DofToQuad::TENSOR;

   // This creates a *copy* of dof2quad, owned.
   dof2quad = fes2.GetFE(0)->GetDofToQuad(fe->GetNodes(), mode);

   // Make copies of the 1D matrices.
   B_1d = dof2quad.B;
   Bt_1d = dof2quad.Bt;
}

void ChangeOfBasis_L2::Mult(const Vector &x, Vector &y) const
{
   NVTX("L2 change basis");
   using namespace internal::quadrature_interpolator;
   dof2quad.B.MakeRef(B_1d);
   TensorValues<QVectorLayout::byVDIM>(ne, 1, dof2quad, x, y);
}

void ChangeOfBasis_L2::MultTranspose(const Vector &x, Vector &y) const
{
   NVTX("L2 change basis transpose");
   using namespace internal::quadrature_interpolator;
   dof2quad.B.MakeRef(Bt_1d);
   TensorValues<QVectorLayout::byVDIM>(ne, 1, dof2quad, x, y);
}

} // namespace mfem
