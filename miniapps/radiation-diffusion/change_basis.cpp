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
#include "fem/qinterp/eval.hpp"

#include "general/nvtx.hpp"

namespace mfem
{

ChangeOfBasis_L2::ChangeOfBasis_L2(FiniteElementSpace &fes1_)
   : Operator(fes1_.GetTrueVSize()),
     fes1(fes1_)
{
   auto *fe = dynamic_cast<const TensorBasisElement*>(fes1.GetFE(0));
   MFEM_VERIFY(fe != NULL, "Must use tensor elements.");
   const Poly_1D::Basis &basis1d = fe->GetBasis1D();
   const int order = fes1.GetOrder(0);
   const int pp1 = order + 1;

   const double *gll_pts = poly1d.GetPoints(order + 1, BasisType::GaussLobatto);
   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, order);
   const int nq = ir.Size();

   Vector b(pp1);
   B1d.SetSize(pp1*pp1);
   B1d = 0.0;
   for (int i = 0; i < pp1; ++i)
   {
      const double h = gll_pts[i+1] - gll_pts[i];
      for (int q = 0; q < nq; ++ q)
      {
         const IntegrationPoint &ip = ir[q];
         const double x = gll_pts[i] + h*ip.x;
         const double w = ip.weight;
         basis1d.Eval(x, b);
         for (int j = 0; j < pp1; ++j)
         {
            B1d[i + j*pp1] += w*b[j];
         }
      }
   }
}

void ChangeOfBasis_L2::Mult(const Vector &x, Vector &y) const
{
   NVTX("L2 change basis");
   const int dim = fes1.GetMesh()->Dimension();
   if (dim == 2) { Mult2D(x, y); }
   else if (dim == 3) { Mult3D(x, y); }
}

void ChangeOfBasis_L2::Mult2D(const Vector &x, Vector &y) const
{
   using namespace internal::quadrature_interpolator;

   const int ne = fes1.GetNE();
   const int order = fes1.GetOrder(0);
   const int pp1 = order + 1;
   const double *X = x.Read();
   double *Y = y.Write();
   const double *B = B1d.Read();

   // vdim = 1, layout doesn't really matter
   constexpr QVectorLayout L = QVectorLayout::byVDIM;
   constexpr int MD = 8;


   // TODO: replace this call by call to TensorValues<QVectorLayout::byVDIM>
   Values2D<L,0,0,0,0,MD,MD>(ne, B, X, Y, 1, pp1, pp1);
}

void ChangeOfBasis_L2::Mult3D(const Vector &x, Vector &y) const
{

}

void ChangeOfBasis_L2::MultTranspose(const Vector &x, Vector &y) const
{
   NVTX("L2 change basis transpose");
   const int dim = fes1.GetMesh()->Dimension();
   if (dim == 2) { MultTranspose2D(x, y); }
   else if (dim == 3) { MultTranspose3D(x, y); }
}

void ChangeOfBasis_L2::MultTranspose2D(const Vector &x, Vector &y) const
{

}

void ChangeOfBasis_L2::MultTranspose3D(const Vector &x, Vector &y) const
{

}

} // namespace mfem
