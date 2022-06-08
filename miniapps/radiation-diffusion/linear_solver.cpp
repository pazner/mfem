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

#include "radiation_diffusion.hpp"
#include "linear_solver.hpp"

namespace mfem
{

RadiationDiffusionLinearSolver::RadiationDiffusionLinearSolver(
   RadiationDiffusionOperator &rad_diff_)
   : Solver(rad_diff_.Height()),
     rad_diff(rad_diff_),
     dt_prev(0.0)
{
   Setup();
}

void RadiationDiffusionLinearSolver::Setup() const
{
   using namespace MMS;

   const double dt = rad_diff.dt;
   // If dt has not changed since last time, we don't need to rebuild the
   // operator or the solver.
   if (dt == dt_prev) { return; }
   dt_prev = dt;

   Array2D<HypreParMatrix*> EF_blocks(2,2);
   Array2D<double> EF_coeff(2,2);

   EF_blocks(0,0) = rad_diff.L.get();
   EF_blocks(0,1) = rad_diff.D.get();
   EF_blocks(1,0) = rad_diff.Dt.get();
   EF_blocks(1,1) = rad_diff.R.get();

   EF_coeff(0,0) = 1.0 + c*dt*sigma;
   EF_coeff(0,1) = 1.0;
   EF_coeff(1,0) = 1.0;
   EF_coeff(1,1) = -3*sigma/c/dt;

   JEF.reset(HypreParMatrixFromBlocks(EF_blocks, &EF_coeff));
   EF_solver.reset(new SerialDirectSolver(*JEF));
}

void RadiationDiffusionLinearSolver::Mult(const Vector &b, Vector &x) const
{
   EF_solver->Mult(b, x);
}

void RadiationDiffusionLinearSolver::SetOperator(const Operator &op) { }

} // namespace mfem
