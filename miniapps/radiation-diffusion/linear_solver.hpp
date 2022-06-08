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
   std::unique_ptr<HypreParMatrix> JEF;
   std::unique_ptr<Solver> EF_solver;
public:
   RadiationDiffusionLinearSolver(class RadiationDiffusionOperator &rad_diff_);
   void Mult(const Vector &b, Vector &x) const override;
   void SetOperator(const Operator &op) override;
};

} // namespace mfem

#endif
