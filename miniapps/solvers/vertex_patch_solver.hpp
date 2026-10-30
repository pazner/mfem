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

#ifndef MFEM_VERTEX_PATCH_SOLVER_HPP
#define MFEM_VERTEX_PATCH_SOLVER_HPP

#include "mfem.hpp"
#include <memory>
#include <vector>

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_LAPACK) && defined(MFEM_USE_METIS) && defined(MFEM_USE_METIS_5)
namespace mfem
{
namespace vertex_patch
{

struct Options
{
   bool static_condensation = true;
   real_t damping = 0.0; ///< Zero selects a power-method estimate.
   real_t diagonal_shift = 0.0; ///< Relative ICC diagonal shift.
   int power_iterations = 30;
};

/// Reference-interval data, cached once per polynomial degree.
struct Basis
{
   int p;
   DenseMatrix S, inverse, G, derivative;
   Vector eigenvalues;
   Array<int> parity;
   explicit Basis(int degree);
   static std::shared_ptr<const Basis> Get(int degree);
};

/// Sum-factorized global FDM-to-GLL conversion, including modal orientations.
class BasisChange : public Operator
{
   ParFiniteElementSpace &fes;
   std::shared_ptr<const Basis> basis;
   int dim, nd;
   Array<int> nodal, modal, signs, nodal_mask, modal_mask;
   mutable Vector local_in, local_out, element, result, scratch1, scratch2;
   void Apply(const Vector &, Vector &, bool inverse, bool transpose) const;
public:
   BasisChange(ParFiniteElementSpace &space,
               std::shared_ptr<const Basis> reference_basis);
   void Mult(const Vector &x, Vector &y) const override { Apply(x, y, false, false); }
   void MultTranspose(const Vector &x, Vector &y) const override
   { Apply(x, y, false, true); }
   void InverseMult(const Vector &x, Vector &y) const { Apply(x, y, true, false); }
   void InverseMultTranspose(const Vector &x, Vector &y) const
   { Apply(x, y, true, true); }
   const Array<int> &ModalMap() const { return modal; }
   const Array<int> &Signs() const { return signs; }
};

/// Sparse element auxiliary operator (3.9), in local lexicographic FDM order.
std::unique_ptr<SparseMatrix> AssembleElement(
   const Basis &basis, ElementTransformation &trans, Coefficient &coefficient);

/// Condense a sparse FDM element with a diagonal interior block.
/// boundary contains the retained lexicographic indices, in increasing order.
std::unique_ptr<SparseMatrix> CondenseElement(
   const SparseMatrix &matrix, int degree, int dimension,
   std::vector<int> &boundary);

/** MPI H1 Poisson preconditioner. The input space must be scalar, conforming,
    uniformly ordered, and use the Gauss-Lobatto basis on quads or hexes.
    The space, mesh, and coefficient must outlive this object. Reconstruct after
    changing mesh, degree, coefficient, or essential boundary conditions.
    Mult is a fixed symmetric linear action on standard H1 true DOFs. */
class Preconditioner : public Solver
{
   struct Impl;
   std::unique_ptr<Impl> impl;
public:
   Preconditioner(ParFiniteElementSpace &space, Coefficient &coefficient,
                  const Array<int> &essential_boundary, const Options &options = {});
   ~Preconditioner() override;
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
   void SetOperator(const Operator &) override
   { MFEM_ABORT("Reconstruct the vertex-patch preconditioner to update it"); }
   void PrintStatistics(std::ostream &os) const;
};

} // namespace vertex_patch
} // namespace mfem
#endif
#endif
