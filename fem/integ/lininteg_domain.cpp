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

#include "../../fem/kernels.hpp"
#include "../../general/forall.hpp"
#include "../fem.hpp"

#include "lininteg_domain_kernels.hpp"

/// \cond DO_NOT_DOCUMENT

namespace mfem
{
static void DLFEvalAssemble(const FiniteElementSpace &fes,
                            const IntegrationRule *ir,
                            const Array<int> &markers, const Vector &coeff,
                            Vector &y)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const FiniteElement &el = *fes.GetTypicalFE();
   const MemoryType mt = Device::GetDeviceMemoryType();
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int d = maps.ndof, q = maps.nqpt;
   constexpr int flags = GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);
   const int map_type = fes.GetTypicalFE()->GetMapType();

   const int vdim = fes.GetVDim();
   const int ne = fes.GetMesh()->GetNE();
   const real_t *B = maps.B.Read();
   const int *M = markers.Read();
   const real_t *detJ = geom->detJ.Read();
   const real_t *W = ir->GetWeights().Read();
   real_t *Y = y.ReadWrite();
   const auto &limits = DeviceDofQuadLimits::Get();
   if (d > limits.MAX_D1D || q > limits.MAX_Q1D)
   {
      int nq = 1;
      for (int axis = 0; axis < dim; ++axis) { nq *= q; }
      Vector u(ne*vdim*nq), v; u.UseDevice(true); v.UseDevice(true);
      const real_t *C = coeff.Read(); const bool constant = coeff.Size() == vdim;
      real_t *U = u.Write();
      mfem::forall(u.Size(), [=] MFEM_HOST_DEVICE (int k)
      {
         const int iq = k%nq, c = k/nq%vdim, e = k/(nq*vdim);
         const real_t det = map_type == FiniteElement::VALUE ? detJ[iq+nq*e] : 1.;
         U[k] = M[e] ? W[iq]*det*C[constant ? c : c+vdim*(iq+nq*e)] : 0.;
      });
      int sizes[3] = {q,q,q};
      for (int axis = 0; axis < dim; ++axis)
      {
         int before = 1, after = 1;
         for (int a = 0; a < axis; ++a) { before *= sizes[a]; }
         for (int a = axis+1; a < dim; ++a) { after *= sizes[a]; }
         const int src_size = before*q*after, dst_size = before*d*after;
         v.SetSize(ne*vdim*dst_size);
         const real_t *src = u.Read(); real_t *dst = v.Write();
         mfem::forall(v.Size(), [=] MFEM_HOST_DEVICE (int k)
         {
            const int batch = k/dst_size, i = k%before;
            const int j = k/before%d, outer = k%dst_size/(before*d);
            real_t value = 0.;
            for (int l = 0; l < q; ++l)
            { value += B[l+q*j]*src[batch*src_size+i+before*(l+q*outer)]; }
            dst[k] = value;
         });
         u.Swap(v); sizes[axis] = d;
      }
      const real_t *result = u.Read();
      mfem::forall(u.Size(), [=] MFEM_HOST_DEVICE (int k) { Y[k] += result[k]; });
      return;
   }
   DomainLFIntegrator::AssembleKernels::Run(dim, d, q, vdim, ne, d, q, map_type,
                                            M, B, detJ, W, coeff, Y);
}

void DomainLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                        const Array<int> &markers, Vector &b)
{
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int qorder = oa * fe.GetOrder() + ob;
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);

   QuadratureSpace qs(*fes.GetMesh(), *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   DLFEvalAssemble(fes, ir, markers, coeff, b);
}

void VectorDomainLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                              const Array<int> &markers,
                                              Vector &b)
{
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int qorder = 2 * fe.GetOrder();
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);

   QuadratureSpace qs(*fes.GetMesh(), *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   DLFEvalAssemble(fes, ir, markers, coeff, b);
}

DomainLFIntegrator::AssembleKernelType
DomainLFIntegrator::AssembleKernels::Fallback(int DIM, int, int)
{
   switch (DIM)
   {
      case 1:
         return DLFEvalAssemble1D<0, 0>;
      case 2:
         return DLFEvalAssemble2D<0, 0>;
      case 3:
         return DLFEvalAssemble3D<0, 0>;
   }
   MFEM_ABORT("");
}

DomainLFIntegrator::Kernels::Kernels()
{
   // 2D
   // Q = P+1
   DomainLFIntegrator::AddSpecialization<2, 1, 1>();
   DomainLFIntegrator::AddSpecialization<2, 2, 2>();
   DomainLFIntegrator::AddSpecialization<2, 3, 3>();
   DomainLFIntegrator::AddSpecialization<2, 4, 4>();
   DomainLFIntegrator::AddSpecialization<2, 5, 5>();
   // Q = P+2
   DomainLFIntegrator::AddSpecialization<2, 2, 3>();
   DomainLFIntegrator::AddSpecialization<2, 3, 4>();
   DomainLFIntegrator::AddSpecialization<2, 4, 5>();
   DomainLFIntegrator::AddSpecialization<2, 5, 6>();
   // 3D
   // Q = P+1
   DomainLFIntegrator::AddSpecialization<3, 1, 1>();
   DomainLFIntegrator::AddSpecialization<3, 2, 2>();
   DomainLFIntegrator::AddSpecialization<3, 3, 3>();
   DomainLFIntegrator::AddSpecialization<3, 4, 4>();
   DomainLFIntegrator::AddSpecialization<3, 5, 5>();
   // Q = P+2
   DomainLFIntegrator::AddSpecialization<3, 2, 3>();
   DomainLFIntegrator::AddSpecialization<3, 3, 4>();
   DomainLFIntegrator::AddSpecialization<3, 4, 5>();
   DomainLFIntegrator::AddSpecialization<3, 5, 6>();
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
