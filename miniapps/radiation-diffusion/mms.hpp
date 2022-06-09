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

#ifndef RAD_DIFF_COEFFICIENTS_HPP
#define RAD_DIFF_COEFFICIENTS_HPP

#include "mfem.hpp"

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

double rad(double x, double y);
double ExactMaterialEnergy(const Vector &xvec, double t);
double ExactRadiationEnergy(const Vector &xvec, double t);
double MaterialEnergySource(const Vector &xvec, double t);
double RadiationEnergySource(const Vector &xvec, double t);

} // namespace MMS

} // namespace mfem

#endif
