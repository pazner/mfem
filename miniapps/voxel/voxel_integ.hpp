// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef VOXEL_INTEG_HPP
#define VOXEL_INTEG_HPP

#include "mfem.hpp"
#include "../../linalg/dtensor.hpp"

namespace mfem
{

class VoxelIntegrator : public BilinearFormIntegrator
{
   std::unique_ptr<BilinearFormIntegrator> integ;
   DenseMatrix elmat;
   int ndof_per_el;
   int ne;
   mutable Vector z;
public:
   VoxelIntegrator(BilinearFormIntegrator *integ_);
   void AssemblePA(const FiniteElementSpace &fes) override;
   void AddMultPA(const Vector&, Vector&) const override;
   void AssembleDiagonalPA(Vector &diag) override;
   const DenseMatrix &GetElementMatrix() const
   {
      // MFEM_VERIFY(elmat.Size() > 0, "");
      return elmat;
   }
};

class VoxelBlockJacobi : public Solver
{
public:
   const int ne;
   const int vdim;
   const int ntdof;
   const int ndof_per_el;
   const bool byvdim;
   Vector blockdiag_tvec;
   const Array<int> &ess_dofs;
public:
   VoxelBlockJacobi(
      ParFiniteElementSpace &fes,
      VoxelIntegrator &integ,
      const Array<int> &ess_dofs_,
      double damping=1.0);
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
   void SetOperator(const Operator &op) override { }
};

class VoxelChebyshev : public Solver
{
private:
   const Operator &op;
   const int order;

   VoxelBlockJacobi block_jacobi;

   double max_eig_estimate;

   Array<double> coeffs;
   mutable Vector r, z;
public:
   VoxelChebyshev(
      const Operator &op_,
      ParFiniteElementSpace &fes,
      VoxelIntegrator &integ,
      const Array<int> &ess_dofs_,
      const int order_);

   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
   void SetOperator(const Operator &op_) override { }
};

}

#endif
