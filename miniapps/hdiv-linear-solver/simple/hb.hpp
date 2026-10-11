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

#ifndef MFEM_HDIV_SIMPLE_HB_HPP
#define MFEM_HDIV_SIMPLE_HB_HPP

#include "mfem.hpp"
#include <memory>

namespace mfem
{

/// Broken RT fluxes and face multipliers for a conforming tensor-product mesh.
/** Broken vectors use the lexicographic element-restriction layout. Multipliers
    label flux continuity constraints on interior faces. Boundary multipliers
    are retained with identity equations. */
class BrokenSpace
{
   ParFiniteElementSpace &fes;
   DG_Interface_FECollection trace_fec;
   ParFiniteElementSpace trace_fes;
   const ElementRestriction &restriction;
   const Operator &prolongation;
   Array<int> inactive_trace;
   Vector inverse_multiplicity;
   std::unique_ptr<HypreParMatrix> Ct, C;

   void Prolongate(const Vector &x, Vector &broken) const;

public:
   explicit BrokenSpace(ParFiniteElementSpace &fes);

   int GetNumBrokenDofs() const;
   int GetNumTraceDofs() const;
   ParFiniteElementSpace &GetFESpace() const;
   HypreParMatrix &GetC();
   HypreParMatrix &GetCt();
   const Array<int> &GetInactiveTraceDofs() const;

   /// Restrict a local conforming grid function to its element copies.
   void LocalProlongate(const Vector &x, Vector &broken) const;
   /// Average element copies into a conforming true-DOF vector.
   void Average(const Vector &broken, Vector &x) const;
   /// Distribute a true-DOF right-hand side among its element copies.
   void AverageTranspose(const Vector &x, Vector &broken) const;
   /// Sum element mass diagonals into true DOFs, without orientation signs.
   void AssembleDiagonal(const Vector &broken_diag, Vector &diag) const;
   /// Map essential true DOFs to their element copies.
   void GetBrokenEssentialDofs(const Array<int> &ess_tdofs,
                               Array<int> &ess_broken_dofs) const;
};

/// Approximate inverse of the multiplier Schur complement C A^{-1} C^T.
/** Applies F^{-1} + F^{-1} E^T S^{-1} E F^{-1}, using diagonal RT and L2
    masses M_b (broken RT), M_g (conforming RT), and W (L2), where
    F = C M_b^{-1} C^T, E = D_hat M_b^{-1} C^T, and
    S = W^{-1} + D M_g^{-1} D^T. D_hat and D are the broken and conforming
    incidence divergences. S^{-1} is approximated by one BoomerAMG cycle.
    Scalar coefficients alpha (mass) and beta (div-div) must be positive. */
class GraphPreconditioner : public Solver
{
   Vector F_inverse;
   std::unique_ptr<HypreParMatrix> F_inverse_Et, S;
   std::unique_ptr<HypreBoomerAMG> S_inv;
   mutable Vector z1, z2;

public:
   GraphPreconditioner(BrokenSpace &fes_hat, const Array<int> &ess_rt_dofs,
                       Coefficient &alpha, Coefficient &beta);
   void SetOperator(const Operator &) override;
   void Mult(const Vector &x, Vector &y) const override;
};

class EssentialDofIdentity : public Operator
{
   Array<int> ess_dofs;
public:
   EssentialDofIdentity(int n, const Array<int> &ess_dofs_)
      : Operator(n), ess_dofs(ess_dofs_) { }
   void Mult(const Vector &x, Vector &y) const
   {
      const auto d_ess_dofs = ess_dofs.Read();
      const auto d_x = x.Read();
      auto d_y = y.Write();
      mfem::forall(height, [=] MFEM_HOST_DEVICE (int i) { d_y[i] = 0.0; });
      mfem::forall(ess_dofs.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         const int j = d_ess_dofs[i];
         d_y[j] = d_x[j];
      });
   }
};

} // namespace mfem

#endif
