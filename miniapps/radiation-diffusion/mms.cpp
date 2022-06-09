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

#include "mms.hpp"
#include "radiation_diffusion.hpp"

namespace mfem
{

namespace MMS
{

double rad(const Vector &x)
{
   const int dim = x.Size();
   if (dim == 1) { return x[0]; }
   else if (dim == 2) { return sqrt(x[0]*x[0] + x[1]*x[1]); }
   else if (dim == 3) { return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]); }
   else { return 0.0; }
}

double ExactMaterialEnergy(const Vector &xvec, double t)
{
   const double r = rad(xvec);
   return Cv * T0 * (1 - 0.5 * exp(-tau * t) * cos(omega * r));
}

double ExactRadiationEnergy(const Vector &xvec, double t)
{
   const double r = rad(xvec);
   const double exponent = exp(-tau * t);
   const double Trad     = T0 * (1 + 0.5 * exponent);
   return a * pow(Trad, 4) * (1 + 0.5 * exponent * cos(omega * r));
}

double MaterialEnergySource(const Vector &xvec, double t)
{
   const double exponent = exp(-tau * t);
   const double cosine   = cos(omega * rad(xvec));
   const double Tmat     = T0 * (1 - 0.5 * exponent * cosine);
   const double Trad     = T0 * (1 + 0.5 * exponent);
   const double E        = a * pow(Trad, 4) * (1 + 0.5 * exponent * cosine);
   return rho * Cv * 0.5 * T0 * tau * exponent * cosine
          +
          c * sigma * (a * pow(Tmat, 4) - E);
}

double RadiationEnergySource(const Vector &xvec, double t)
{
   const double r = rad(xvec);
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

} // namespace MMS

} // namespace mfem