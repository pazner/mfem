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

#include "general/forall.hpp"

namespace mfem
{

LinearizedMaterialEnergyOperator::LinearizedMaterialEnergyOperator(
   MaterialEnergyOperator &H_)
   : Operator(H_.fes.GetTrueVSize()),
     H(H_),
     qf(&H.qs),
     coeff(qf),
     mass_integrator(coeff),
     x_q(H.qs.GetSize())
{
   mass_integrator.SetIntegrationRule(H.qs.GetElementIntRule(0));
}

void LinearizedMaterialEnergyOperator::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   mass_integrator.AddMultPA(x, y);
}

void LinearizedMaterialEnergyOperator::SetLinearizationState(
   const Vector &x) const
{
   using namespace MMS;

   const int ne = H.fes.GetMesh()->GetNE();
   const int nq_per_el = H.qs.GetElementIntRule(0).Size();
   const double DT = H.dt;

   H.qinterp.Values(x, x_q);

   const double *d_detJ = H.geom->detJ.Read();
   const double *d_e = H.e_q.Read();
   const double *d_k = x_q.Read();
   double *d_qf = qf.Write();

   MFEM_FORALL(ii, ne*nq_per_el,
   {
      const double det_J = d_detJ[ii];
      const double e_val = d_e[ii]/det_J;
      const double k_val = d_k[ii]/det_J;
      const double e_np1 = e_val + DT*k_val;
      const double ans = 4*a*c*eta*sigma*DT*pow(Cv, -4)*pow(e_np1, 3);
      d_qf[ii] = ans;
   });


   mass_integrator.AssemblePA(H.fes);
}

void LinearizedMaterialEnergyOperator::AssembleDiagonal(Vector &diag) const
{
   diag = 0.0;
   mass_integrator.AssembleDiagonalPA(diag);
}

int GetQuadratureOrder(FiniteElementSpace &fes)
{
   return 2*fes.GetMaxElementOrder() + fes.GetElementTransformation(0)->OrderW();
}

MaterialEnergyOperator::MaterialEnergyOperator(FiniteElementSpace &fes_)
   : Operator(fes_.GetTrueVSize()),
     fes(fes_),
     qs(fes.GetMesh(), GetQuadratureOrder(fes)),
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
   lf_integrator.SetIntRule(&qs.GetElementIntRule(0));
   markers = 1;
}

void MaterialEnergyOperator::SetMaterialEnergy(const Vector &e_gf) const
{
   qinterp.Values(e_gf, e_q);
}

void MaterialEnergyOperator::Mult(const Vector &x, Vector &y) const
{
   using namespace MMS;

   const int ne = fes.GetMesh()->GetNE();
   const int nq_per_el = qs.GetElementIntRule(0).Size();

   qinterp.Values(x, x_q);

   const double *d_detJ = geom->detJ.Read();
   const double *d_e = e_q.Read();
   const double *d_k = x_q.Read();
   double *d_qf = qf.Write();

   const double DT = dt; // Assign to local so we don't access *this from kernel

   MFEM_FORALL(ii, ne*nq_per_el,
   {
      const double det_J = d_detJ[ii];
      const double e_val = d_e[ii]/det_J;
      const double k_val = d_k[ii]/det_J;

      const double e_np1 = e_val + DT*k_val;
      const double T = e_np1/Cv;
      const double ans = a*c*eta*sigma*pow(T, 4);

      d_qf[ii] = ans;
   });

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

} // namespace mfem
