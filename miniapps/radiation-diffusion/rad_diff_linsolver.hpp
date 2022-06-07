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

#ifndef RAD_DIFF_LINSOLVER_HPP
#define RAD_DIFF_LINSOLVER_HPP

#include "mfem.hpp"
#include "rad_diff_coefficients.hpp"
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
   T4DerivativeCoefficient lin_coeff;
   ParBilinearForm dH_form;
   BlockOperator J;

   std::unique_ptr<HypreParMatrix> dH, J00, JeE, JEF;
   std::unique_ptr<Solver> eE_solver, EF_solver;

   std::unique_ptr<HypreParMatrix> JJ;
   std::unique_ptr<Solver> J_solver;

   mutable ParGridFunction ke_star, kE_star, F_star;
   mutable Vector r, c_eE, c_EF;

protected:
   double ComputeResidual() const;
   void SolveEnergies() const;

public:
   RadiationDiffusionLinearSolver(class RadiationDiffusionOperator &rad_diff_);
   void Mult(const Vector &b, Vector &x) const override;
   void SetOperator(const Operator &op) override;
   void Update(const Vector &x);
};

} // namespace mfem

#endif
