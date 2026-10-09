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

#ifndef MFEM_HDIV_GENERIC_COEFFICIENT_HPP
#define MFEM_HDIV_GENERIC_COEFFICIENT_HPP

#include "mfem.hpp"

namespace mfem
{

/// A scalar or diagonal matrix coefficient. A default-constructed (or null)
/// GenericCoefficient is the unit coefficient.
struct GenericCoefficient
{
   Coefficient *Q = nullptr;
   DiagonalMatrixCoefficient *D = nullptr;
   GenericCoefficient() = default;
   GenericCoefficient(Coefficient &Q_) : Q(&Q_) { }
   GenericCoefficient(DiagonalMatrixCoefficient &D_) : D(&D_) { }
   GenericCoefficient(Coefficient *Q_) : Q(Q_) { }
   GenericCoefficient(DiagonalMatrixCoefficient *D_) : D(D_) { }

   /// Number of components: the dimension for a diagonal matrix coefficient,
   /// one otherwise.
   int GetVDim() const { return D ? D->GetVDim() : 1; }

   /// Return a new integrator of type INTEGRATOR with this coefficient.
   template <typename INTEGRATOR>
   INTEGRATOR *NewIntegrator() const
   {
      if (Q) { return new INTEGRATOR(Q); }
      else if (D) { return new INTEGRATOR(D); }
      else { return new INTEGRATOR; }
   }
};

} // namespace mfem

#endif
