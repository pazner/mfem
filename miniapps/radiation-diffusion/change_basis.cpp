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
   const IntegrationRule &ir = fes1.GetFE(0)->GetNodes();
   const auto mode = DofToQuad::TENSOR;

   // NOTE: this assumes that fes1 uses a *nodal basis*
   // This creates a *copy* of dof2quad.
   dof2quad = fes2.GetFE(0)->GetDofToQuad(ir, mode);

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

ChangeOfBasis_RT::ChangeOfBasis_RT(FiniteElementSpace &fes1,
                                   FiniteElementSpace &fes2)
   : Operator(fes1.GetTrueVSize()),
     ne(fes1.GetNE())
{
   const FiniteElementCollection *fec = fes1.FEColl();
   const auto *rt_fec = dynamic_cast<const RT_FECollection*>(fec);
   MFEM_VERIFY(rt_fec != NULL, "Must be RC finite element collection.");

   const int cb_type = rt_fec->GetClosedBasisType();
   const int ob_type = rt_fec->GetOpenBasisType();

   const int p = fes1.GetMaxElementOrder();
   const int pp1 = p + 1;

   const double *cpts1 = poly1d.GetPoints(p, cb_type);
   const double *opts1 = poly1d.GetPoints(p - 1, ob_type);

   const auto &cb2 = poly1d.GetBasis(p, BasisType::GaussLobatto);
   const auto &ob2 = poly1d.GetBasis(p - 1, BasisType::IntegratedGLL);

   Vector b;

   // CofB maps from interp-histop to nodal
   // ith nodal point

   // Evaluate cb2 at cb1
   Bc_1d.SetSize(pp1*pp1);
   Bct_1d.SetSize(pp1*pp1);
   b.SetSize(pp1);
   for (int i = 0; i < pp1; ++i)
   {
      cb2.Eval(cpts1[i], b);
      for (int j = 0; j < pp1; ++j)
      {
         Bc_1d[i + j*pp1] = b[j];
         Bct_1d[j + i*pp1] = b[j];
      }
   }

   // Evaluate ob2 at ob1
   Bo_1d.SetSize(p*p);
   Bot_1d.SetSize(p*p);
   b.SetSize(p);
   for (int i = 0; i < p; ++i)
   {
      ob2.Eval(opts1[i], b);
      for (int j = 0; j < p; ++j)
      {
         Bo_1d[i + j*p] = b[j];
         Bot_1d[j + i*p] = b[j];
      }
   }
}

void ChangeOfBasis_RT::Mult(const Vector &x, Vector &y) const
{
   NVTX("RT change basis");
}

void ChangeOfBasis_RT::MultTranspose(const Vector &x, Vector &y) const
{
   NVTX("RT change basis transpose");
}

} // namespace mfem
