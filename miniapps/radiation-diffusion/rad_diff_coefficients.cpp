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

#include "rad_diff_coefficients.hpp"
#include "rad_diff_operator.hpp"

namespace mfem
{

namespace MMS
{

double rad(double x, double y)
{
   return sqrt(x*x + y*y);
}

double InitialMaterialEnergy(const Vector &xvec)
{
   const double x = xvec[0], y = xvec[1];
   return Cv*T0*(1 - 0.5*cos(omega * rad(x,y)));
}

double InitialRadiationEnergy(const Vector &xvec)
{
   const double x = xvec[0], y = xvec[1];
   return a*pow(T0*1.5, 4)*(1 + 0.5 * cos(omega * rad(x,y)));
}

double ExactMaterialEnergy(const Vector &xvec, double t)
{
   const double x = xvec[0], y = xvec[1];
   const double r = rad(x,y);
   return Cv * T0 * (1 - 0.5 * exp(-tau * t) * cos(omega * r));
}

double ExactRadiationEnergy(const Vector &xvec, double t)
{
   const double x = xvec[0], y = xvec[1];
   const double r = rad(x,y);
   const double exponent = exp(-tau * t);
   const double Trad     = T0 * (1 + 0.5 * exponent);
   return a * pow(Trad, 4) * (1 + 0.5 * exponent * cos(omega * r));
}

double MaterialEnergySource(const Vector &xvec, double t)
{
   const double x = xvec[0], y = xvec[1];

   const double exponent = exp(-tau * t);
   const double cosine   = cos(omega * rad(x,y));
   const double Tmat     = T0 * (1 - 0.5 * exponent * cosine);
   const double Trad     = T0 * (1 + 0.5 * exponent);
   const double E        = a * pow(Trad, 4) * (1 + 0.5 * exponent * cosine);
   return rho * Cv * 0.5 * T0 * tau * exponent * cosine
          +
          c * sigma * (a * pow(Tmat, 4) - E);
}

double RadiationEnergySource(const Vector &xvec, double t)
{
   const double x = xvec[0], y = xvec[1];
   const double r = rad(x,y);
   const double exponent = exp(-tau * t);
   const double cosine   = cos(omega * r);
   const double Tmat     = T0 * (1 - 0.5 * exponent * cosine);
   const double Trad     = T0 * (1 + 0.5 * exponent);
   const double E        = a * pow(Trad, 4) * (1 + 0.5 * exponent * cosine);
   return -0.5 * tau * exponent * a * pow(Trad, 3) *
          (4 * T0 + (Trad + 2 * T0 * exponent) * cosine)
          +
          c * exponent * a * pow(Trad, 4) / (6 * sigma) *
          (omega*omega * cosine + omega * sin(omega * r) / r)
          -
          c * sigma * (a * pow(Tmat, 4) - E);
}

} // namespace MMS

T4Coefficient::T4Coefficient(RadiationDiffusionOperator &rad_diff_)
   : rad_diff(rad_diff_), b_gf(&rad_diff.fes_l2) { }

double T4Coefficient::Eval(ElementTransformation &Tr,
                           const IntegrationPoint &ip)
{
   using namespace MMS;

   const double dt = rad_diff.dt;

   const double e_val = rad_diff.e_gf.GetValue(Tr, ip);
   const double k_val = b_gf.GetValue(Tr, ip);
   const double e_np1 = e_val + dt*k_val;
   const double T = e_np1/Cv;

   return c*eta*sigma*pow(T, 4);
   // return c*eta*sigma*T;
   // return T;
   // return e_val/Cv + dt*k_val/Cv;
}

T4DerivativeCoefficient::T4DerivativeCoefficient(
   RadiationDiffusionOperator &rad_diff_)
   : rad_diff(rad_diff_), b_gf(&rad_diff.fes_l2) { }

double T4DerivativeCoefficient::Eval(ElementTransformation &Tr,
                                     const IntegrationPoint &ip)
{
   using namespace MMS;

   const double dt = rad_diff.dt;

   const double e_val = rad_diff.e_gf.GetValue(Tr, ip);
   const double k_val = b_gf.GetValue(Tr, ip);
   const double e_np1 = e_val + dt*k_val;

   return 4*c*eta*sigma*dt*pow(Cv, -4)*pow(e_np1, 3);
   // return c*eta*sigma*dt/Cv;
   // return dt/Cv;
   // return dt/Cv;
}

} // namespace mfem
