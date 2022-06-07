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

#ifndef RAD_DIFF_OPERATOR_HPP
#define RAD_DIFF_OPERATOR_HPP

#include "mfem.hpp"
#include "rad_diff_linsolver.hpp"
#include "rad_diff_coefficients.hpp"
#include <memory>

namespace mfem
{

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

class RadiationDiffusionOperator : public TimeDependentOperator
{
   friend class NonlinearEnergyIntegrator;
   friend class RadiationDiffusionLinearSolver;
   friend class T4Coefficient;
   friend class T4DerivativeCoefficient;

   // private:
public:
   static constexpr int b1 = BasisType::GaussLobatto;
   static constexpr int b2 = BasisType::IntegratedGLL;
   static constexpr int mt = FiniteElement::INTEGRAL;

   const int dim; // Spatial dimension

   L2_FECollection fec_l2;
   ParFiniteElementSpace fes_l2; // Finite element space for energies

   RT_FECollection fec_rt;
   ParFiniteElementSpace fes_rt; // Finite element space for flux

   ParGridFunction e_gf, E_gf; // Material and radiation energy
   ParGridFunction F_gf; // Radiation flux
   ParGridFunction ke_gf, kE_gf; // Material and radiation DIRK increments

   FunctionCoefficient S_e_coeff, S_E_coeff; // Source term coefficients

   ParNonlinearForm H_form;
   ParBilinearForm L_form;
   ParBilinearForm R_form;
   ParMixedBilinearForm D_form;

   std::unique_ptr<HypreParMatrix> L;
   std::unique_ptr<HypreParMatrix> D, Dt;
   std::unique_ptr<HypreParMatrix> R;

   std::unique_ptr<RadiationDiffusionLinearSolver> linear_solver;
   NewtonSolver newton;

   Array<int> offsets;

   double dt;

   mutable Vector b, z;

public:
   RadiationDiffusionOperator(ParMesh &mesh, int order);
   double GetTimeStep() const { return dt; }
   void Mult(const Vector &x, Vector &y) const override;
   void ImplicitSolve(const double dt_, const Vector &x, Vector &k) override;
   Operator &GetGradient(const Vector &x) const override
   {
      const Operator &op = *this;
      return const_cast<Operator&>(op);
   }
   const Array<int> &GetOffsets() const { return offsets; }
   ParFiniteElementSpace &GetL2Space() { return fes_l2; }
   ParFiniteElementSpace &GetRTSpace() { return fes_rt; }
};

} // namespace mfem

#endif
