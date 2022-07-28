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

#ifndef RAD_DIFF_MMS_HPP
#define RAD_DIFF_MMS_HPP

#include "mfem.hpp"
#include "general/forall.hpp"

namespace mfem
{

namespace MMS
{

// For problem specification, see the paper
//
// [1] T. A. Brunner, Development of a grey nonlinear thermal radiation
//     diffusion verification problem (2006). SAND2006-4030C.

// For the definition of the constants, see Table I from reference [1].
static constexpr double rho   = 2;
static constexpr double Cv    = 3;
static constexpr double sigma = 4;
static constexpr double T0    = 1e5;

static constexpr double c     = 2.99792458e+8;
static constexpr double a     = 7.56576651e-16;
static constexpr double tau   = 2.27761040e+9;
static constexpr double omega = 2.13503497e+1;

static constexpr double eta = 1;

// We define the coefficients of the problem as HOST_DEVICE functions, so that
// they can be evaluated both on CPU and GPU

MFEM_HOST_DEVICE inline
double rad(int dim, const double *x)
{
   if (dim == 1) { return x[0]; }
   else if (dim == 2) { return sqrt(x[0]*x[0] + x[1]*x[1]); }
   else if (dim == 3) { return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]); }
   else { return 0.0; }
}

MFEM_HOST_DEVICE inline
double ExactMaterialEnergy(int dim, const double *xvec, double t)
{
   const double r = rad(dim, xvec);
   return Cv * T0 * (1 - 0.5 * exp(-tau * t) * cos(omega * r));
}

MFEM_HOST_DEVICE inline
double ExactRadiationEnergy(int dim, const double *xvec, double t)
{
   const double r = rad(dim, xvec);
   const double exponent = exp(-tau * t);
   const double Trad     = T0 * (1 + 0.5 * exponent);
   return a * pow(Trad, 4) * (1 + 0.5 * exponent * cos(omega * r));
}

MFEM_HOST_DEVICE inline
double MaterialEnergySource(int dim, const double *xvec, double t)
{
   const double exponent = exp(-tau * t);
   const double cosine   = cos(omega * rad(dim, xvec));
   const double Tmat     = T0 * (1 - 0.5 * exponent * cosine);
   const double Trad     = T0 * (1 + 0.5 * exponent);
   const double E        = a * pow(Trad, 4) * (1 + 0.5 * exponent * cosine);
   return rho * Cv * 0.5 * T0 * tau * exponent * cosine
          +
          c * sigma * (a * pow(Tmat, 4) - E);
}

MFEM_HOST_DEVICE inline
double RadiationEnergySource(int dim, const double *xvec, double t)
{
   const double r = rad(dim, xvec);
   const double exponent = exp(-tau * t);
   const double cosine   = cos(omega * r);
   const double Tmat     = T0 * (1 - 0.5 * exponent * cosine);
   const double Trad     = T0 * (1 + 0.5 * exponent);
   const double E        = a * pow(Trad, 4) * (1 + 0.5 * exponent * cosine);
   return -0.5 * tau * exponent * a * pow(Trad, 3) *
          (4 * T0 + (Trad + 2 * T0 * exponent) * cosine)
          + c * exponent * a * pow(Trad, 4) / (6 * sigma) *
          (omega*omega * cosine + omega * sin(omega * r) / r)
          - c * sigma * (a * pow(Tmat, 4) - E);
}

using FunctionType = double(*)(int dim, const double *x, double t);

// Wrap the HOST_DEVICE functions in a form that can be used by
// FunctionCoefficient.
template <FunctionType F>
double CoefficientFunction(const Vector &xvec, double t)
{
   return F(xvec.Size(), xvec.GetData(), t);
}
static constexpr auto ExactMaterialEnergyFunction =
   CoefficientFunction<ExactMaterialEnergy>;
static constexpr auto ExactRadiationEnergyFunction =
   CoefficientFunction<ExactRadiationEnergy>;

// This class contains the coefficients needed for the time evolution of the
// problem. The coefficients are of type QuadratureFunctionCoefficient, and the
// coefficient functions are evaluated at the quadrature functions in a kernel
// that can run in parallel on device.
class Coefficients
{
private:
   QuadratureSpace qs;
   const IntegrationRule &ir;
   QuadratureFunction Q_e_qf;
   QuadratureFunction S_E_qf;
   const GeometricFactors *geom;
public:
   QuadratureFunctionCoefficient Q_e;
   QuadratureFunctionCoefficient S_E;

   Coefficients(Mesh &mesh, int order)
      : qs(&mesh, 2*(order-1)),
        ir(qs.GetElementIntRule(0)),
        Q_e_qf(&qs),
        S_E_qf(&qs),
        Q_e(Q_e_qf),
        S_E(S_E_qf)
   {
      geom = mesh.GetGeometricFactors(ir, GeometricFactors::COORDINATES);
      SetTime(0.0);
   }
   template <FunctionType F>
   void SetTime(QuadratureFunctionCoefficient &coeff, double t)
   {
      coeff.SetTime(t);
      auto &qf = const_cast<QuadratureFunction&>(coeff.GetQuadFunction());
      const int dim = qs.GetMesh()->Dimension();
      const int nq = ir.Size();
      const int n = qf.Size();
      const double T = t;
      const double *x = geom->X.Read();
      double *q = qf.Write();

      MFEM_FORALL(ii, n,
      {
         const int i = ii / nq;
         const int j = ii % nq;
         double xvec[3];
         for (int d = 0; d < dim; ++d)
         {
            xvec[d] = x[j + d*nq + i*dim*nq];
         }
         q[ii] = F(dim, xvec, T);
      });
   }
   void SetTime(double t)
   {
      SetTime<MaterialEnergySource>(Q_e, t);
      SetTime<RadiationEnergySource>(S_E, t);
   }
   const IntegrationRule &GetIntRule() { return ir; }
};

} // namespace MMS

} // namespace mfem

#endif
