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

#ifndef CHANGE_BASIS_HPP
#define CHANGE_BASIS_HPP

#include "mfem.hpp"
#include <memory>

namespace mfem
{

/// @brief Change of basis operator from given L2 space to IntegratedGLL basis.
class ChangeOfBasis_L2 : public Operator
{
private:
   const int ne; ///< Number of elements in the mesh.
   mutable DofToQuad dof2quad; ///< 1D basis transformation.
   Array<double> B_1d; ///< 1D basis transformation matrix.
   Array<double> Bt_1d; ///< 1D basis transformation matrix traspose.
public:
   ChangeOfBasis_L2(FiniteElementSpace &fes1, FiniteElementSpace &fes2);
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override;
};

/// @brief Change of basis operator from given RT space to IntegratedGLL basis.
class ChangeOfBasis_RT : public Operator
{
private:
   const int ne;
   mutable DofToQuad dof2quad_closed; ///< 1D closed basis transformation.
   Array<double> Bc_1d; ///< 1D closed basis transformation matrix.
   Array<double> Bct_1d; ///< 1D closed basis transformation matrix transpose.
   mutable DofToQuad dof2quad_open; ///< 1D open basis transformation.
   Array<double> Bo_1d; ///< 1D open basis transformation matrix.
   Array<double> Bot_1d; ///< 1D open basis transformation matrix transpose.
public:
   ChangeOfBasis_RT(FiniteElementSpace &fes1, FiniteElementSpace &fes2);
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override;
};

} // namespace mfem

#endif
