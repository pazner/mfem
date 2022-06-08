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
#include "linear_solver.hpp"

namespace mfem
{

class RadiationDiffusionOperator;

class NonlinearEnergyOperator : public Operator
{
private:
   RadiationDiffusionOperator &rad_diff;
   mutable Vector z;
   mutable std::unique_ptr<HypreParMatrix> J, J00;
public:
   NonlinearEnergyOperator(RadiationDiffusionOperator &rad_diff_);
   void Mult(const Vector &x, Vector &y) const override;
   Operator &GetGradient(const Vector &x) const override;
};

class BrunnerNowackIteration : public Solver
{
private:
   RadiationDiffusionOperator &rad_diff;
   mutable Vector c_eE, c_EF, r, z;

   NonlinearEnergyOperator N_eE;
   NewtonSolver eE_solver;
   SerialDirectSolver J_eE_solver;
   RadiationDiffusionLinearSolver EF_solver;

   void ApplyFullOperator(const Vector &x, Vector &y) const;

public:
   BrunnerNowackIteration(RadiationDiffusionOperator &rad_diff_);
   void Mult(const Vector &b, Vector &x) const override;
   void SetOperator(const Operator &op) override;
};

} // namespace mfem

#endif
