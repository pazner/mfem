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

#ifndef NONLINEAR_ITERATION_HPP
#define NONLINEAR_ITERATION_HPP

#include "mfem.hpp"
#include "../hdiv-linear-solver/hdiv_linear_solver.hpp"
#include "energy_integrator.hpp"

namespace mfem
{

class RadiationDiffusionOperator;

/// Class to compute the action of the linearized operator
///
/// [ rho*L + dH         -c*dt*sigma*L    ]
/// [     -dH          (1 + c*dt*sigma)*L ]
///
/// This is the linearization/Jacobian of NonlinearEnergyOperator.
class LinearizedEnergyOperator : public Operator
{
   friend class EnergyBlockJacobi;
   RadiationDiffusionOperator &rad_diff;
   mutable Vector z;
   double dt;
   const Operator *dH;
public:
   LinearizedEnergyOperator(RadiationDiffusionOperator &rad_diff_);
   void Mult(const Vector &x, Vector &y) const;
   void SetTimeStep(const double dt_) { dt = dt_; }
   void SetLinearizedMaterialOperator(const Operator &dH_) { dH = &dH_; }
};

// Class to compute the action of the nonlinear operator
// [ rho*L + H         -c*dt*sigma*L    ]
// [     -H          (1 + c*dt*sigma)*L ]
class NonlinearEnergyOperator : public Operator
{
   friend class LinearizedEnergyOperator;
private:
   RadiationDiffusionOperator &rad_diff;
   mutable Vector z;
   mutable LinearizedEnergyOperator linearized_op;
   double dt;
public:
   NonlinearEnergyOperator(RadiationDiffusionOperator &rad_diff_);
   void Mult(const Vector &x, Vector &y) const override;
   Operator &GetGradient(const Vector &x) const override;
   void Setup(const double dt);
};

class EnergyBlockJacobi : public Solver
{
private:
   Vector block_jacobi;
   Vector diag_dH;
public:
   void Mult(const Vector &x, Vector &y) const;
   void SetOperator(const Operator &op);
};

class RadiationDiffusionLinearSolver : public Solver
{
private:
   ConstantCoefficient L_coeff, R_coeff;
   HdivSaddlePointLinearSolver solver;
   double dt_prev;
public:
   RadiationDiffusionLinearSolver(RadiationDiffusionOperator &rad_diff_);
   void SetOperator(const Operator &op) { }
   void Mult(const Vector &x, Vector &y) const { solver.Mult(x, y); }
   void Setup(double dt);
   int GetNumIterations() const { return solver.GetNumIterations(); }
};

class BrunnerNowackIteration : public IterativeSolver
{
private:
   RadiationDiffusionOperator &rad_diff;
   mutable Vector c_eE, c_EF, r, z;

   NonlinearEnergyOperator N_eE;
   GMRESSolver J_eE_solver;
   EnergyBlockJacobi J_eE_preconditioner;
   NewtonSolver eE_solver;
   ConstantCoefficient L_coeff, R_coeff;
   RadiationDiffusionLinearSolver EF_solver;

   void ApplyFullOperator(const Vector &x, Vector &y) const;

public:
   BrunnerNowackIteration(RadiationDiffusionOperator &rad_diff_);
   void Mult(const Vector &b, Vector &x) const override;
   void SetOperator(const Operator &op) override;
   void Setup(const double dt);
};

} // namespace mfem

#endif
