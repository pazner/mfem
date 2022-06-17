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

   const double ans = a*c*eta*sigma*pow(T, 4);

   return ans;
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

   return 4*a*c*eta*sigma*dt*pow(Cv, -4)*pow(e_np1, 3);
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

MaterialEnergyOperator::MaterialEnergyOperator(FiniteElementSpace &fes_)
   : Operator(fes_.GetTrueVSize()),
     fes(fes_),
     qs(fes.GetMesh(), 2*fes.GetMaxElementOrder()),
     qinterp(fes, qs),
     qf(&qs),
     coeff(qf),
     lf_integrator(coeff),
     geom(fes.GetMesh()->GetGeometricFactors(qs.GetElementIntRule(0),
                                             GeometricFactors::DETERMINANTS)),
     linearized_op(*this),
     e_q(qs.GetSize()),
     x_q(qs.GetSize()),
     markers(fes.GetMesh()->GetNE()),
     dt(0.0)
{
   markers = 1;
}

void MaterialEnergyOperator::SetMaterialEnergy(const Vector &e_gf) const
{
   qinterp.Values(e_gf, e_q);
}

void MaterialEnergyOperator::Mult(const Vector &x, Vector &y) const
{
   using namespace MMS;

   const int nq_per_el = qs.GetElementIntRule(0).Size();

   qinterp.Values(x, x_q);

   Vector qf_vals;
   for (int e = 0; e < fes.GetMesh()->GetNE(); ++e)
   {
      const IntegrationRule &ir = qs.GetElementIntRule(e);
      qf.GetElementValues(e, qf_vals);
      for (int i = 0; i < ir.Size(); ++i)
      {
         const double det_J = geom->detJ[i + e*nq_per_el];
         const double e_val = e_q[i + e*nq_per_el]/det_J;
         const double k_val = x_q[i + e*nq_per_el]/det_J;
         const double e_np1 = e_val + dt*k_val;
         const double T = e_np1/Cv;
         const double ans = a*c*eta*sigma*pow(T, 4);
         qf_vals[i] = ans;
      }
   }

   y = 0.0;
   lf_integrator.AssembleDevice(fes, markers, y);
}

LinearizedMaterialEnergyOperator &MaterialEnergyOperator::GetLinearizedOperator(
   const Vector &x)
{
   linearized_op.SetLinearizationState(x);
   return linearized_op;
}

Operator &MaterialEnergyOperator::GetGradient(const Vector &x) const
{
   linearized_op.SetLinearizationState(x);
   const Operator &op = linearized_op;
   return const_cast<Operator&>(op);
}

LinearizedMaterialEnergyOperator::LinearizedMaterialEnergyOperator(
   MaterialEnergyOperator &H_)
   : Operator(H_.fes.GetTrueVSize()),
     H(H_),
     qf(&H.qs),
     coeff(qf),
     mass_integrator(coeff),
     x_q(H.qs.GetSize())
{ }

void LinearizedMaterialEnergyOperator::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   mass_integrator.AddMultPA(x, y);
}

void LinearizedMaterialEnergyOperator::SetLinearizationState(
   const Vector &x) const
{
   using namespace MMS;

   const int nq_per_el = H.qs.GetElementIntRule(0).Size();
   const double dt = H.dt;

   H.qinterp.Values(x, x_q);

   Vector qf_vals;
   for (int e = 0; e < H.fes.GetMesh()->GetNE(); ++e)
   {
      const IntegrationRule &ir = H.qs.GetElementIntRule(e);
      qf.GetElementValues(e, qf_vals);
      for (int i = 0; i < ir.Size(); ++i)
      {
         const double det_J = H.geom->detJ[i + e*nq_per_el];
         const double e_val = H.e_q[i + e*nq_per_el]/det_J;
         const double k_val = x_q[i + e*nq_per_el]/det_J;
         const double e_np1 = e_val + dt*k_val;
         const double ans = 4*a*c*eta*sigma*dt*pow(Cv, -4)*pow(e_np1, 3);
         qf_vals[i] = ans;
      }
   }

   mass_integrator.AssemblePA(H.fes);
}

} // namespace mfem
