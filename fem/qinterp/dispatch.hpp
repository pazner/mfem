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

// Internal header, included only by .cpp files

#include "../quadinterpolator.hpp"
#include "eval.hpp"

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

// Tensor-product evaluation of quadrature point values: dispatch function.
template<QVectorLayout VL>
void HdivTensorValues(const int NE,
                      const int vdim,
                      const GeometricFactors *geom,
                      const DofToQuad &maps,
                      const DofToQuad &maps_o,
                      const Vector &e_vec,
                      Vector &q_val)
{
   if (NE == 0) { return; }
   const int dim = maps.FE->GetDim();
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const double *Bc = maps.B.Read();
   const double *Bo = maps_o.B.Read();
   const double *X = e_vec.Read();
   const double *J = geom->J.Read();
   const double *detJ = geom->detJ.Read();
   double *Y = q_val.Write();

   constexpr QVectorLayout L = VL;

   if (dim == 2)
   {
      constexpr int MD = MAX_D1D;
      constexpr int MQ = MAX_Q1D;
      MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                  << " are not supported!");
      MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
                  << MQ << " 1D points are not supported!");
      HdivValues2D<L,MD,MQ>(NE,J,detJ,Bc,Bo,X,Y,vdim,D1D,Q1D);
   }
   else if (dim == 3)
   {
      constexpr int MD = 8;
      constexpr int MQ = 8;
      MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                  << " are not supported!");
      MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
                  << MQ << " 1D points are not supported!");
      HdivValues3D<L,MD,MQ>(NE,J,detJ,Bc,Bo,X,Y,vdim,D1D,Q1D);
   }
   else
   {
      MFEM_ABORT("Dimension not supported");
   }
}

// Tensor-product evaluation of quadrature point values: dispatch function.
template<QVectorLayout VL>
void TensorValues(const int NE,
                  const int vdim,
                  const DofToQuad &maps,
                  const Vector &e_vec,
                  Vector &q_val);

// Tensor-product evaluation of quadrature point derivatives: dispatch function.
template<QVectorLayout VL>
void TensorDerivatives(const int NE,
                       const int vdim,
                       const DofToQuad &maps,
                       const Vector &e_vec,
                       Vector &q_der);

// Tensor-product evaluation of quadrature point physical derivatives: dispatch
// function.
template<QVectorLayout VL>
void TensorPhysDerivatives(const int NE,
                           const int vdim,
                           const DofToQuad &maps,
                           const GeometricFactors &geom,
                           const Vector &e_vec,
                           Vector &q_der);

// Tensor-product evaluation of quadrature point determinants: dispatch
// function.
void TensorDeterminants(const int NE,
                        const int vdim,
                        const DofToQuad &maps,
                        const Vector &e_vec,
                        Vector &q_det,
                        Vector &d_buff);

} // namespace quadrature_interpolator

} // namespace internal

} // namespace mfem
