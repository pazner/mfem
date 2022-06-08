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
   friend class NonlinearEnergyOperator;
   friend class RadiationDiffusionLinearSolver;
   friend class BrunnerNowackIteration;
   friend class T4Coefficient;
   friend class T4DerivativeCoefficient;

   // TODO:
private:
   static constexpr int b1 = BasisType::GaussLobatto; ///< "closed basis"
   static constexpr int b2 = BasisType::IntegratedGLL; ///< "open basis"

   const int dim; ///< Spatial dimension.

   L2_FECollection fec_l2; ///< L2 collection.
   ParFiniteElementSpace fes_l2; ///< L2 space for material and radiation energy.

   RT_FECollection fec_rt; ///< RT collection.
   ParFiniteElementSpace fes_rt; ///< RT space for radiation flux.

   ParGridFunction e_gf; ///< Material energy, needed for H integrator

   ParNonlinearForm H_form; ///< Nonlinear energy term.
   ParBilinearForm L_form; ///< L2 mass matrix.
   ParBilinearForm R_form; ///< RT mass matrix.
   ParMixedBilinearForm D_form; ///< RT -> L2 divergence.

   FunctionCoefficient Q_e_coeff; ///< Material energy source coefficient.
   FunctionCoefficient S_E_coeff; ///< Radiation energy source coefficient.
   FunctionCoefficient E_bdr_coeff; ///< Radiation energy boundary condition.

   ParLinearForm Q_e; ///< Material energy source term.
   ParLinearForm S_E; ///< Radiation energy source term.
   ParLinearForm b_n; ///< Radiation energy boundary term (in flux equation).

   std::unique_ptr<HypreParMatrix> L; ///< Assembled L2 mass matrix.
   std::unique_ptr<HypreParMatrix> D; ///< Assembled divergence form.
   std::unique_ptr<HypreParMatrix> Dt; ///< The transpose of @ref D.
   std::unique_ptr<HypreParMatrix> R; ///< Assembled RT mass matrix.

   /// Brunner-Nowack nonlinear (outer) iterative solver.
   std::unique_ptr<BrunnerNowackIteration> nonlinear_solver;

   Array<int> offsets;

   double dt; ///< Time step.

   mutable Vector b; ///< Right-hand side for nonlinear solve.
   mutable Vector z; ///< Used as a temporary vector for computations.

public:
   /// Construct the radiation-diffusion operator given @a mesh and @a order.
   RadiationDiffusionOperator(ParMesh &mesh, int order);
   /// Solve the system k = f(x + dt*k), needed for DIRK-type time integration.
   void ImplicitSolve(const double dt_, const Vector &x, Vector &k) override;
   /// Set the current time, update the source terms.
   void SetTime(const double t_) override;
   /// Get the offsets array for the block vector of unknowns.
   const Array<int> &GetOffsets() const { return offsets; }
   /// Get the L2 space used for the material and radiation energies.
   ParFiniteElementSpace &GetL2Space() { return fes_l2; }
   /// Get the RT space used for the radiation flux.
   ParFiniteElementSpace &GetRTSpace() { return fes_rt; }
   /// Return the associated MPI communicator
   MPI_Comm GetComm() const { return fes_l2.GetComm(); }
};

} // namespace mfem

#endif
