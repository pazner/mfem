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

#ifndef LINEAR_SOLVER_HPP
#define LINEAR_SOLVER_HPP

#include "mfem.hpp"
#include "mms.hpp"
#include "change_basis.hpp"
#include <memory>

namespace mfem
{

class RadiationDiffusionLinearSolver : public Solver
{
private:
   MINRESSolver minres;

   static constexpr int b1 = BasisType::GaussLobatto;
   static constexpr int b2 = BasisType::IntegratedGLL;
   static constexpr int mt = FiniteElement::INTEGRAL;

   const int order;

   // L2 and RT spaces, using the interpolation-histopolation bases
   L2_FECollection fec_l2;
   ParFiniteElementSpace fes_l2;

   RT_FECollection fec_rt;
   ParFiniteElementSpace fes_rt;

   // Change of basis operators
   ChangeOfBasis_L2 basis_l2;
   ChangeOfBasis_RT basis_rt;

   ParBilinearForm mass_rt;

   // Components needed for the block operator
   OperatorHandle R;
   std::unique_ptr<HypreParMatrix> D, Dt;
   std::unique_ptr<Operator> L_inv;
   Vector L_diag;

   // Components needed for the preconditioner
   OperatorJacobiSmoother R_inv;
   HypreBoomerAMG S_inv;

   std::unique_ptr<HypreParMatrix> S;

   Array<int> offsets, ess_dofs;
   std::unique_ptr<BlockOperator> A_block;
   std::unique_ptr<BlockDiagonalPreconditioner> D_prec;

   ConstantCoefficient R_coeff;

   double dt_prev;

   mutable Vector b_prime, x_prime, z;
public:
   RadiationDiffusionLinearSolver(ParMesh &mesh,
                                  ParFiniteElementSpace &fes_rt_,
                                  ParFiniteElementSpace &fes_l2_);
   /// Build the linear operator and solver. Must be called when dt changes.
   void Setup(const double dt);
   /// Solve the linear system for material and radiation energy.
   void Mult(const Vector &b, Vector &x) const override;
   /// No-op.
   void SetOperator(const Operator &op) override;
   /// Get the number of MINRES iterations.
   int GetNumIterations() const { return minres.GetNumIterations(); }
};

} // namespace mfem

#endif
