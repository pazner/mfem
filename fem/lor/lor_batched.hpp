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

#ifndef MFEM_LOR_BATCHED
#define MFEM_LOR_BATCHED

#include "lor.hpp"

namespace mfem
{

/// @brief Efficient batched assembly of LOR discretizations on device.
///
/// This class should typically be used by the user-facing classes
/// LORDiscretization and ParLORDiscretization. Only certain bilinear forms are
/// supported, currently:
///
///  - H1 diffusion + mass
///  - ND curl-curl + mass (2D only)
class BatchedLORAssembly
{
protected:
   FiniteElementSpace &fes_ho; ///< The high-order space.

   Vector X_vert; ///< LOR vertex coordinates.

   /// Get the vertices of the LOR mesh and place the result in @a X_vert.
   template <int Q1D> void GetLORVertexCoordinates();

   /// @brief The elementwise LOR matrices in a sparse "ij" format.
   ///
   /// This is interpreted to have shape (nnz_per_row, ndof_per_el, nel_ho). For
   /// index (i, j, k), this represents row @a j of the @a kth element matrix.
   /// The column index is given by sparse_mapping(i, j).
   Vector sparse_ij;

   /// @brief The sparsity pattern of the element matrices.
   ///
   /// For local DOF index @a j, sparse_mapping(i, j) is the column index of the
   /// @a ith nonzero in the @a jth row. If the index is negative, that entry
   /// should be skipped (there is no corresponding nonzero).
   DenseMatrix sparse_mapping;

public:
   /// Does the given form support batched assembly?
   virtual bool FormIsSupported(BilinearForm &a) = 0;

   /// @brief Create a BatchedLORAssembly object that supports assembling the
   /// given form.
   static BatchedLORAssembly *New(BilinearForm &a, FiniteElementSpace &fes_ho);

   /// Assemble the system, and place the result in @a A.
   void Assemble(BilinearForm &a, const Array<int> &ess_dofs, OperatorHandle &A);

   virtual ~BatchedLORAssembly() { }
protected:
   virtual void SetForm(BilinearForm &a) = 0;

   /// After assembling the "sparse IJ" format, convert it to CSR.
   void SparseIJToCSR(OperatorHandle &A) const;

   /// Assemble the system without eliminating essential DOFs.
   void AssembleWithoutBC(OperatorHandle &A);

   /// Called by one of the specialized classes, e.g. BatchedLORDiffusion.
   BatchedLORAssembly(FiniteElementSpace &fes_ho_);

   /// Return the first domain integrator in the form @a i of type @a T.
   template <typename T>
   static T *GetIntegrator(BilinearForm &a)
   {
      Array<BilinearFormIntegrator*> *integs = a.GetDBFI();
      if (integs != NULL)
      {
         for (auto *i : *integs)
         {
            if (auto *ti = dynamic_cast<T*>(i))
            {
               return ti;
            }
         }
      }
      return nullptr;
   }

   // Compiler limitation: these should be protected, but they contain
   // MFEM_FORALL kernels, and so they must be public.
public:
   /// @brief Fill the I array of the sparse matrix @a A.
   ///
   /// @note AssemblyKernel must be called first to populate @a sparse_mapping.
   int FillI(SparseMatrix &A) const;

   /// @brief Fill the J and data arrays of the sparse matrix @a A.
   ///
   /// @note AssemblyKernel must be called first to populate @a sparse_mapping
   /// and @a sparse_ij.
   void FillJAndData(SparseMatrix &A) const;

#ifdef MFEM_USE_MPI
   /// Assemble the system in parallel and place the result in @a A.
   void ParAssemble(const Array<int> &ess_dofs, OperatorHandle &A);
#endif

   /// @brief Pure virtual function for the kernel actually performing the
   /// assembly. Overridden in the derived classes.
   virtual void AssemblyKernel() = 0;
};

template <typename T1, typename T2>
bool HasIntegrators(BilinearForm &a)
{
   Array<BilinearFormIntegrator*> *integs = a.GetDBFI();
   if (integs == NULL) { return false; }
   if (integs->Size() == 1)
   {
      BilinearFormIntegrator *i0 = (*integs)[0];
      if (dynamic_cast<T1*>(i0) || dynamic_cast<T2*>(i0)) { return true; }
   }
   else if (integs->Size() == 2)
   {
      BilinearFormIntegrator *i0 = (*integs)[0];
      BilinearFormIntegrator *i1 = (*integs)[1];
      if ((dynamic_cast<T1*>(i0) && dynamic_cast<T2*>(i1)) ||
          (dynamic_cast<T2*>(i0) && dynamic_cast<T1*>(i1)))
      {
         return true;
      }
   }
   return false;
}

}

#endif
