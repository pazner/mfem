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

#ifndef RADIATION_DIFFUSION_HPP
#define RADIATION_DIFFUSION_HPP

#include "mfem.hpp"
#include "radiation_diffusion.hpp"
#include "nonlinear_iteration.hpp"
#include <memory>

namespace mfem
{

class RadiationDiffusionOperator : public TimeDependentOperator
{
   friend class NonlinearEnergyIntegrator;
   friend class RadiationDiffusionLinearSolver;
   friend class T4Coefficient;
   friend class T4DerivativeCoefficient;

   // TODO:
   // private:
public:
   static constexpr int b1 = BasisType::GaussLobatto;
   static constexpr int b2 = BasisType::IntegratedGLL;
   static constexpr int mt = FiniteElement::INTEGRAL;

   const int dim; // Spatial dimension

   // Finite element space for material and radiation energy
   L2_FECollection fec_l2;
   ParFiniteElementSpace fes_l2;

   // Finite element space for radiation flux
   RT_FECollection fec_rt;
   ParFiniteElementSpace fes_rt;

   ParGridFunction e_gf; // Material energy, needed for H integrator
   FunctionCoefficient S_e_coeff, S_E_coeff; // Source term coefficients

   ParNonlinearForm H_form; // Nonlinear energy term
   ParBilinearForm L_form; // L2 mass matrix
   ParBilinearForm R_form; // RT mass matrix
   ParMixedBilinearForm D_form; // RT -> L2 divergence

   // Assembled matrices
   std::unique_ptr<HypreParMatrix> L;
   std::unique_ptr<HypreParMatrix> D, Dt;
   std::unique_ptr<HypreParMatrix> R;

   std::unique_ptr<BrunnerNowackIteration> nonlinear_solver;

   Array<int> offsets;

   double dt;

   mutable Vector b, z;

public:
   RadiationDiffusionOperator(ParMesh &mesh, int order);
   void ImplicitSolve(const double dt_, const Vector &x, Vector &k) override;
};

} // namespace mfem

#endif
