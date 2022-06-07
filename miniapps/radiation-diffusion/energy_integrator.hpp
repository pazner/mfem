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

#ifndef ENERGY_INTEGRATOR_HPP
#define ENERGY_INTEGRATOR_HPP

#include "mfem.hpp"
#include "mms.hpp"
#include <memory>

namespace mfem
{

class T4Coefficient : public Coefficient
{
   class RadiationDiffusionOperator &rad_diff;
public:
   ParGridFunction b_gf;
   T4Coefficient(class RadiationDiffusionOperator &rad_diff_);
   double Eval(ElementTransformation &Tr, const IntegrationPoint &ip);
};

class T4DerivativeCoefficient : public Coefficient
{
   class RadiationDiffusionOperator &rad_diff;
public:
   ParGridFunction b_gf;
   T4DerivativeCoefficient(class RadiationDiffusionOperator &rad_diff_);
   double Eval(ElementTransformation &Tr, const IntegrationPoint &ip);
};

class NonlinearEnergyIntegrator : public NonlinearFormIntegrator
{
   ParFiniteElementSpace &fes;
   T4Coefficient coeff;
   DomainLFIntegrator integ;

   T4DerivativeCoefficient deriv_coeff;
   MassIntegrator mass;
   Array<int> dofs;

public:
   NonlinearEnergyIntegrator(class RadiationDiffusionOperator &rad_diff_);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun, DenseMatrix &elmat);
};

} // namespace mfem

#endif
