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
   // TODO: delete these friends
   friend class NonlinearEnergyIntegrator;
   friend class RadiationDiffusionLinearSolver;
   friend class T4Coefficient;
   friend class T4DerivativeCoefficient;

   // TODO:
   // private:
public:
   static constexpr int b1 = BasisType::GaussLobatto; ///< "closed basis"
   static constexpr int b2 = BasisType::IntegratedGLL; ///< "open basis"

   const int dim; ///< Spatial dimension.

   L2_FECollection fec_l2; ///< L2 collection.
   ParFiniteElementSpace fes_l2; ///< L2 space for material and radiation energy.

   RT_FECollection fec_rt; ///< RT collection.
   ParFiniteElementSpace fes_rt; ///< RT space for radiation flux.

   ParGridFunction e_gf; ///< Material energy, needed for H integrator
   FunctionCoefficient S_e_coeff, S_E_coeff; // Source term coefficients

   ParNonlinearForm H_form; ///< Nonlinear energy term.
   ParBilinearForm L_form; ///< L2 mass matrix.
   ParBilinearForm R_form; ///< RT mass matrix.
   ParMixedBilinearForm D_form; ///< RT -> L2 divergence.

   std::unique_ptr<HypreParMatrix> L; ///< Assembled L2 mass matrix.
   std::unique_ptr<HypreParMatrix> D; ///< Assembled divergence form.
   std::unique_ptr<HypreParMatrix> Dt; ///< The transpose of @ref D.
   std::unique_ptr<HypreParMatrix> R; ///< Assembled RT mass matrix.

   /// Brunner-Nowack nonlinear (outer) iterative solver.
   std::unique_ptr<BrunnerNowackIteration> nonlinear_solver;

   Array<int> offsets;

   // TODO: propagate the time step to the solvers/nonlinear operators
   double dt; ///< Time step.

   mutable Vector b; ///< Right-hand side for nonlinear solve.
   mutable Vector z; ///< Used as a temporary vector for computations.

public:
   RadiationDiffusionOperator(ParMesh &mesh, int order);
   void ImplicitSolve(const double dt_, const Vector &x, Vector &k) override;
};

} // namespace mfem

#endif
