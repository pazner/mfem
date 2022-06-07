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

#include "energy_integrator.hpp"
#include "radiation_diffusion.hpp"

namespace mfem
{

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

NonlinearEnergyIntegrator::NonlinearEnergyIntegrator(
   RadiationDiffusionOperator &rad_diff)
   : fes(rad_diff.fes_l2), coeff(rad_diff), integ(coeff),
     deriv_coeff(rad_diff), mass(deriv_coeff) { }

void NonlinearEnergyIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr,
   const Vector &elfun, Vector &elvect)
{
   fes.GetElementDofs(Tr.ElementNo, dofs);
   coeff.b_gf.SetSubVector(dofs, elfun);
   integ.AssembleRHSElementVect(el, Tr, elvect);
}

void NonlinearEnergyIntegrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   fes.GetElementDofs(Tr.ElementNo, dofs);
   deriv_coeff.b_gf.SetSubVector(dofs, elfun);
   mass.AssembleElementMatrix(el, Tr, elmat);
}

} // namespace mfem
