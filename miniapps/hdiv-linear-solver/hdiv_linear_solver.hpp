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
#include "change_basis.hpp"
#include <memory>

namespace mfem
{

enum class L2CoefficientMode
{
   IDENTITY,
   RECIPROCAL
};

/// @brief Solve the saddle-point system using MINRES with block diagonal
/// preconditioning.
class HdivSaddlePointLinearSolver : public Solver
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

   ParBilinearForm mass_l2, mass_rt;

   const Array<int> &ess_rt_dofs;

   // Components needed for the block operator
   OperatorHandle R;
   std::unique_ptr<HypreParMatrix> D, Dt;
   std::unique_ptr<DGMassInverse> L_inv;
   Vector L_diag, R_diag;

   // Components needed for the preconditioner
   OperatorJacobiSmoother R_inv;
   HypreBoomerAMG S_inv;

   std::unique_ptr<HypreParMatrix> S;

   Array<int> offsets;
   std::unique_ptr<BlockOperator> A_block;
   std::unique_ptr<BlockDiagonalPreconditioner> D_prec;

   Coefficient &L_coeff, &R_coeff;

   L2CoefficientMode coeff_mode;
   QuadratureSpace qs;
   QuadratureFunction qf;
   QuadratureFunctionCoefficient L_inv_coeff;

   mutable Vector b_prime, x_prime, z;
public:
   HdivSaddlePointLinearSolver(ParMesh &mesh_,
                               ParFiniteElementSpace &fes_rt_,
                               ParFiniteElementSpace &fes_l2_,
                               Coefficient &L_coeff_,
                               Coefficient &R_coeff_,
                               const Array<int> &ess_rt_dofs_,
                               L2CoefficientMode coeff_mode_ = L2CoefficientMode::IDENTITY);

   /// @brief Build the linear operator and solver. Must be called when the
   /// coefficients change.
   void Setup();
   /// Solve the linear system for material and radiation energy.
   void Mult(const Vector &b, Vector &x) const override;
   /// No-op.
   void SetOperator(const Operator &op) override;
   /// Get the number of MINRES iterations.
   int GetNumIterations() const { return minres.GetNumIterations(); }
};

// TEMPORARY: REMOVE ME
HypreParMatrix *DiagonalInverse(
   Vector &diag_vec, const ParFiniteElementSpace &fes);

} // namespace mfem

#endif
