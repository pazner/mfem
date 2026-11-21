// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MINIAPPS_TETLOR_HPP
#define MFEM_MINIAPPS_TETLOR_HPP

#include "mfem.hpp"
#include <memory>

namespace tetlor
{
using namespace mfem;

inline void PrintProblemSize(int p, int q, long long high_dofs,
                             long long lor_dofs)
{
   out << '\n'
       << "p = " << p << '\n'
       << "q = " << q << '\n'
       << "HO DOFs  = " << high_dofs << '\n'
       << "LOR DOFs = " << lor_dofs << "\n\n";
}

/// LOR-to-Bernstein transfer on true DOFs with homogeneous Dirichlet data.
class ElementTransfer : public Operator
{
   std::unique_ptr<Operator> local;
   OperatorPtr transfer;

public:
   /// Set up the local fits and assemble the physical LOR diffusion form.
   ElementTransfer(FiniteElementSpace &high, BilinearForm &lor_form,
                   int q, const std::string &grid_type, bool reference_fit);

   void Mult(const Vector &x, Vector &y) const override { transfer->Mult(x, y); }
   void MultTranspose(const Vector &x, Vector &y) const override
   { transfer->MultTranspose(x, y); }
};

class FictitiousSpaceSolver : public Solver
{
   const Operator &R, &aux_inv;
   const Array<int> &ess_dofs;
   mutable Vector rhs, sol;

public:
   FictitiousSpaceSolver(const Operator &R_, const Operator &aux_inv_,
                         const Array<int> &ess_dofs_)
      : Solver(R_.Height()), R(R_), aux_inv(aux_inv_),
        ess_dofs(ess_dofs_), rhs(R.Width()), sol(R.Width()) { }
   void SetOperator(const Operator &op) override { }
   void Mult(const Vector &x, Vector &y) const override
   {
      R.MultTranspose(x, rhs);
      aux_inv.Mult(rhs, sol);
      R.Mult(sol, y);
      for (int i : ess_dofs) { y(i) = x(i); }
   }
};

} // namespace tetlor
#endif
