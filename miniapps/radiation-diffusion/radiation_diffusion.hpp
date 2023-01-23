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
#include "mms.hpp"

#include <memory>

namespace mfem
{

class RadiationDiffusionOperator : public TimeDependentOperator
{
   friend class LinearizedEnergyOperator;
   friend class NonlinearEnergyOperator;
   friend class EnergyBlockJacobi;
   friend class RadiationDiffusionLinearSolver;
   friend class BrunnerNowackIteration;
private:
   static constexpr int b1 = BasisType::GaussLobatto; ///< "closed basis"
   static constexpr int b2 = BasisType::GaussLegendre; ///< "open basis"

   ParMesh &mesh; ///< The underlying mesh.

   const int dim; ///< Spatial dimension.

   L2_FECollection fec_l2; ///< L2 collection.
   ParFiniteElementSpace fes_l2; ///< L2 space for material and radiation energy.

   RT_FECollection fec_rt; ///< RT collection.
   ParFiniteElementSpace fes_rt; ///< RT space for radiation flux.

   ParGridFunction e_gf; ///< Material energy, needed for H integrator

   MaterialEnergyOperator H; ///< Nonlinear energy term.
   ParBilinearForm L_form; ///< L2 mass matrix.
   ParBilinearForm R_form; ///< RT mass matrix.
   ParMixedBilinearForm D_form; ///< RT -> L2 divergence.

   MMS::Coefficients coeffs; ///< Coefficients associated with problem.

   FunctionCoefficient E_bdr_coeff; ///< Radiation energy boundary condition.

   ParLinearForm Q_e_form; ///< Material energy source term.
   ParLinearForm S_E_form; ///< Radiation energy source term.
   ParLinearForm b_n_form; ///< Radiation energy boundary term (in flux eqn).

   Vector Q_e; ///< True-dof version of Q_e_form.
   Vector S_E; ///< True-dof version of S_E_form.
   Vector b_n; ///< True-dof version of b_n_form.

   OperatorHandle L; ///< L2 mass matrix.
   OperatorHandle D; ///< Assembled divergence form.
   OperatorHandle Dt; ///< The transpose of @ref D.
   OperatorHandle R; ///< Assembled RT mass matrix.

   Vector diag_L; ///< Diagonal of the L2 mass matrix.

   /// Brunner-Nowack nonlinear (outer) iterative solver.
   std::unique_ptr<BrunnerNowackIteration> nonlinear_solver;

   Array<int> offsets; ///< Offsets for the block vector [e, E, F].

   double dt; ///< Time step.

   mutable Vector b; ///< Right-hand side for nonlinear solve.
   mutable Vector z; ///< Used as a temporary vector for computations.

   Array<int> empty; ///< Needed for some operations with no essential DOFs.

public:
   /// Construct the radiation-diffusion operator given @a mesh and @a order.
   RadiationDiffusionOperator(ParMesh &mesh_, int order);
   /// Solve the system k = f(x + dt*k), needed for DIRK-type time integration.
   void ImplicitSolve(const double dt_, const Vector &x, Vector &k) override;
   /// Set the current time, update the source terms.
   void SetTime(const double t_) override;
   /// Get the offsets array for the block vector of unknowns.
   const Array<int> &GetOffsets() const { return offsets; }
   /// Get the mesh (const version).
   const ParMesh &GetMesh() const { return mesh; }
   /// Get the mesh (non-const version).
   ParMesh &GetMesh() { return mesh; }
   /// Get the L2 space used for the material and radiation energies.
   ParFiniteElementSpace &GetL2Space() { return fes_l2; }
   /// Get the RT space used for the radiation flux.
   ParFiniteElementSpace &GetRTSpace() { return fes_rt; }
   /// Return the associated MPI communicator
   MPI_Comm GetComm() const { return fes_l2.GetComm(); }
   /// Compute the radiation flux, given material and radiation energy
   void ComputeFlux(Vector &x) const;
   /// Return the device memory class, do work on GPU if possible.
   MemoryClass GetMemoryClass() const override { return Device::GetMemoryClass(); }
};

} // namespace mfem

#endif
