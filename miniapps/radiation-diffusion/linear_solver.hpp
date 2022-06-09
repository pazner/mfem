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
#include <memory>

namespace mfem
{

struct SerialDirectSolver : Solver
{
   SparseMatrix diag;
   UMFPackSolver solver;

   SerialDirectSolver() { }
   SerialDirectSolver(HypreParMatrix &A)
   {
      A.GetDiag(diag);
      solver.SetOperator(diag);
   }
   void Mult(const Vector &x, Vector &y) const
   {
      solver.Mult(x, y);
   }
   void SetOperator(const Operator &A_)
   {
      if (auto *A = dynamic_cast<const HypreParMatrix*>(&A_))
      {
         A->GetDiag(diag);
         solver.SetOperator(diag);
      }
      else
      {
         MFEM_ABORT("Must be a HypreParMatrix.");
      }
   }
};

class RadiationDiffusionLinearSolver : public Solver
{
private:
   class RadiationDiffusionOperator &rad_diff;
   MINRESSolver minres;

   static constexpr int b1 = BasisType::GaussLobatto;
   static constexpr int b2 = BasisType::IntegratedGLL;
   static constexpr int mt = FiniteElement::INTEGRAL;

   // L2 and RT spaces, using the interpolation-histopolation bases
   L2_FECollection fec_l2;
   ParFiniteElementSpace fes_l2;

   RT_FECollection fec_rt;
   ParFiniteElementSpace fes_rt;

   // Change of basis operators
   ParDiscreteLinearOperator basis_l2, basis_rt;
   std::unique_ptr<HypreParMatrix> B_l2, B_rt;

   ParBilinearForm mass_rt;

   // Components needed for the block operator
   OperatorHandle R;
   std::unique_ptr<HypreParMatrix> D, Dt;
   std::unique_ptr<Operator> L_inv;

   // Components needed for the preconditioner
   OperatorJacobiSmoother R_inv;
   HypreBoomerAMG S_inv;

   std::unique_ptr<HypreParMatrix> S;

   Array<int> offsets, ess_dofs;
   std::unique_ptr<BlockOperator> A_block;
   std::unique_ptr<BlockDiagonalPreconditioner> D_prec;

   ConstantCoefficient L_coeff, R_coeff;

   double dt_prev;

   mutable Vector b_prime, x_prime, z;
public:
   RadiationDiffusionLinearSolver(class RadiationDiffusionOperator &rad_diff_);
   /// Build the linear operator and solver. Must be called when dt changes.
   void Setup();
   /// Solve the linear system for material and radiation energy.
   void Mult(const Vector &b, Vector &x) const override;
   /// No-op.
   void SetOperator(const Operator &op) override;
   /// Get the number of MINRES iterations.
   int GetNumIterations() const { return minres.GetNumIterations(); }
};

} // namespace mfem

#endif
