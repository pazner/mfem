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

// Computes the action of the linearized operator dH(k_e), linearized about
// a given state.
class LinearizedMaterialEnergyOperator : public Operator
{
   class MaterialEnergyOperator &H;
   mutable QuadratureFunction qf;
   QuadratureFunctionCoefficient coeff;
   mutable MassIntegrator mass_integrator;
   mutable Vector x_q; // Linearization state x evaluated at quadrature points
public:
   LinearizedMaterialEnergyOperator(class MaterialEnergyOperator &H_);

   void SetLinearizationState(const Vector &x) const;

   void Mult(const Vector &x, Vector &y) const override;

   void AssembleDiagonal(Vector &diag) const override;
};

// Computes the action of the operator H(k_e)
class MaterialEnergyOperator : public Operator
{
   friend class LinearizedMaterialEnergyOperator;
private:
   FiniteElementSpace &fes;
   QuadratureSpace qs;
   QuadratureInterpolator qinterp;
   mutable QuadratureFunction qf;
   QuadratureFunctionCoefficient coeff;
   mutable DomainLFIntegrator lf_integrator;
   const GeometricFactors *geom;
   LinearizedMaterialEnergyOperator linearized_op;

   mutable Vector e_q; // Material energy evaluated at quadrature points
   mutable Vector x_q; // Input vector x evaluated at quadrature points

   Array<int> markers; // Needed for lf_integrator.AssembleDevice

   double dt;
public:
   MaterialEnergyOperator(FiniteElementSpace &fes_);

   void SetMaterialEnergy(const Vector &e_gf) const;

   void Mult(const Vector &x, Vector &y) const override;

   void SetTimeStep(const double dt_) { dt = dt_; }

   LinearizedMaterialEnergyOperator &GetLinearizedOperator(const Vector &x);

   Operator &GetGradient(const Vector &x) const override;
};

} // namespace mfem

#endif
