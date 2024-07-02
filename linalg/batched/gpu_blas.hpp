// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_GPU_BLAS_LINALG
#define MFEM_GPU_BLAS_LINALG

#include "batched.hpp"
#include "../../general/backends.hpp"

#ifdef MFEM_USE_CUDA_OR_HIP

#if defined(MFEM_USE_CUDA)
#include <cublas.h>
#elif defined(MFEM_USE_HIP)
#include <hipblas/hipblas.h>
#endif

namespace mfem
{

class GPUBlasBatchedLinAlg : public BatchedLinAlgBase
{
public:
   void Mult(const DenseTensor &A, const Vector &x, Vector &y) const override;
   void Invert(DenseTensor &A) const override;
   void LUFactor(DenseTensor &A, Array<int> &P) const override;
   void LUSolve(const DenseTensor &LU, const Array<int> &P,
                Vector &x) const override;
};

/// Singleton class represented a cuBLAS or hipBLAS handle.
class GPUBlas
{
#if defined(MFEM_USE_CUDA)
   using HandleType = cublasHandle_t;
#elif defined(MFEM_USE_HIP)
   using HandleType = hipblasHandle_t;
#endif

   HandleType handle = nullptr; ///< The internal handle.
   GPUBlas(); ///< Create the handle.
   ~GPUBlas(); ///< Destroy the handle.
   static GPUBlas &Instance(); ///< Get the unique instnce.
public:
   /// Return the handle, creating it if needed.
   static HandleType Handle();
   /// Enable atomic operations.
   static void EnableAtomics();
   /// Disable atomic operations.
   static void DisableAtomics();
};

} // namespace mfem

#endif // MFEM_USE_CUDA_OR_HIP

#endif