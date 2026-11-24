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

#include "lor_mesh.hpp"
#include "general/forall.hpp"
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

/// One reference stencil and a conforming patch-node layout on the coarse mesh.
class BatchedLOR
{
   std::unique_ptr<FiniteElementCollection> collection;
   std::unique_ptr<FiniteElementSpace> space;
   Mesh patch;
   std::vector<BarycentricLabel> labels;
   Array<int> dofs, rows, columns;
   Array2D<int> vertex_permutations;
   DenseMatrix stencil, metrics;
   Vector volumes;

public:
   BatchedLOR(Mesh &mesh, int q, const std::string &grid_type);
   /// Coarse topology space for numbering, boundary DOFs and prolongation.
   FiniteElementSpace &Space() const { return *space; }
   Mesh &ReferencePatch() { return patch; }
   const auto &PatchLabels() const { return labels; }
   const int *VertexPermutation(int e) const { return vertex_permutations[e]; }
   const DenseMatrix &PatchMetrics() const { return metrics; }
   const Vector &PatchVolumes() const { return volumes; }
   const Array<int> &PatchDofs() const { return dofs; }

   /// Compute Q^t K for a physical patch, or the reference patch if e < 0.
   void ApplyPatchStiffness(int e, const DenseMatrix &Q, DenseMatrix &QtK) const;
   void ApplyPatchStiffness(const real_t metric[6], const DenseMatrix &Q,
                            DenseMatrix &QtK) const;
   void GetPatchStiffness(const real_t metric[6], DenseMatrix &K) const;
   void GetPatchStiffness(int e, DenseMatrix &K) const;
   /// Assemble the unconstrained rank-local matrix in an empty CSR matrix.
   void AssembleLocal(SparseMatrix &matrix) const;
   void FormSystemMatrix(const Array<int> &ess_dofs, OperatorHandle &A) const;
};

enum class TransferMethod { Fit, Entity, Interp };
TransferMethod GetTransferMethod(const std::string &name);

/// Sample the high-order basis in reference patch vertex order.
void SamplePatchBasis(const Mesh &patch, const FiniteElement &fe,
                      DenseMatrix &V);

struct TransferType
{
   TransferMethod method;
   bool reference;
};

/// LOR-to-Bernstein transfer on true DOFs with homogeneous Dirichlet data.
class ElementTransfer : public Operator
{
   std::unique_ptr<Operator> local;
   OperatorPtr transfer;

public:
   MemoryClass GetMemoryClass() const override { return Device::GetMemoryClass(); }
   /// Build the LOR-to-high-order transfer for the selected method and geometry.
   ElementTransfer(FiniteElementSpace &high, BatchedLOR &lor,
                   TransferType type);

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
        ess_dofs(ess_dofs_), rhs(R.Width(), Device::GetDeviceMemoryType()),
        sol(R.Width(), Device::GetDeviceMemoryType())
   {
      rhs.UseDevice(true);
      sol.UseDevice(true);
   }
   MemoryClass GetMemoryClass() const override { return R.GetMemoryClass(); }
   void SetOperator(const Operator &op) override { }
   void Mult(const Vector &x, Vector &y) const override
   {
      R.MultTranspose(x, rhs);
      aux_inv.Mult(rhs, sol);
      R.Mult(sol, y);
      if (ess_dofs.Size())
      {
         const auto dofs = ess_dofs.Read();
         const auto input = x.Read();
         auto output = y.ReadWrite();
         mfem::forall(ess_dofs.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            output[dofs[i]] = input[dofs[i]];
         });
      }
   }
};

} // namespace tetlor
#endif
