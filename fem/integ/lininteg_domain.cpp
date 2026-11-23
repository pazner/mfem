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
   DomainLFIntegrator::AssembleKernels::Run(dim, d, q, vdim, ne, d, q, map_type,
                                            M, B, detJ, W, coeff, Y);
}

// Transpose the three one-dimensional Bernstein basis actions used by simplex
// mass PA, starting directly from the weighted load at the Stroud points.
static void AssembleBernsteinTetLoad(const H1Pos_TetrahedronElement &fe,
                                     const IntegrationRule &ir, int ne,
                                     const Array<int> &markers,
                                     const Vector &weighted, Vector &b)
{
   const auto &maps = static_cast<const RaggedDofToQuad&>(
                         fe.GetDofToQuad(ir, DofToQuad::RAGGED_TENSOR));
   const int d = maps.ndof, q = maps.nqpt, nd = fe.GetDof();
   const int npair = d*(d+1)/2;
   const auto B1 = Reshape(maps.Ba1.Read(), d, q);
   const auto B2 = Reshape(maps.Ba2.Read(), npair, q);
   const auto B3 = Reshape(maps.Ba3.Read(), nd, q);
   const auto pair_indices = Reshape(maps.inverse_map2d_mass.Read(), 2, npair);
   const auto indices = Reshape(maps.inverse_map3d_mass.Read(), 2, nd);
   const auto pair_number = Reshape(maps.forward_map2d_mass.Read(), d, d);
   const auto F = Reshape(weighted.Read(), q, q, q, ne);
   const auto active = markers.Read();
   const auto mt = Device::GetDeviceMemoryType();
   Vector first(d*q*q*ne, mt), second(npair*q*ne, mt);
   const auto F1 = Reshape(first.Write(), d, q, q, ne);
   mfem::forall(d*q*q, ne, [=] MFEM_HOST_DEVICE (int k, int e)
   {
      if (!active[e]) { return; }
      const int a = k%d, j = (k/d)%q, l = k/(d*q);
      real_t sum = 0.0;
      for (int i = 0; i < q; ++i) { sum += B1(a,i)*F(i,j,l,e); }
      F1(a,j,l,e) = sum;
   });
   const auto F2 = Reshape(second.Write(), npair, q, ne);
   mfem::forall(npair*q, ne, [=] MFEM_HOST_DEVICE (int k, int e)
   {
      if (!active[e]) { return; }
      const int pair = k%npair, a = pair_indices(0,pair), l = k/npair;
      real_t sum = 0.0;
      for (int j = 0; j < q; ++j) { sum += B2(pair,j)*F1(a,j,l,e); }
      F2(pair,l,e) = sum;
   });
   const auto Y = Reshape(b.ReadWrite(), nd, ne);
   mfem::forall(nd, ne, [=] MFEM_HOST_DEVICE (int a, int e)
   {
      if (!active[e]) { return; }
      const int pair = pair_number(indices(1,a),indices(0,a));
      real_t sum = 0.0;
      for (int l = 0; l < q; ++l) { sum += B3(a,l)*F2(pair,l,e); }
      Y(a,e) += sum;
   });
}

static void AssembleTetDomainLF(const FiniteElementSpace &fes,
                                const H1Pos_TetrahedronElement &fe,
                                const IntegrationRule &ir, bool default_rule,
                                Coefficient &coeff, const Array<int> &markers,
                                Vector &b)
{
   Mesh &mesh = *fes.GetMesh();
   const int ne = mesh.GetNE(), nd = fe.GetDof(), nq = ir.GetNPoints();
   const auto active = markers.Read();
   const auto mt = Device::GetDeviceMemoryType();
   const bool stroud = !default_rule &&
                       &ir == &StroudIntRules.Get(fe.GetGeomType(), ir.GetOrder());

   // On affine tetrahedra every degree-p Bernstein basis function integrates
   // to volume/nd. This is the exact reference B^T W for a constant load.
   const GridFunction *nodes = mesh.GetNodes();
   const auto *constant = dynamic_cast<ConstantCoefficient*>(&coeff);
   if (constant && (default_rule || stroud) && ir.GetOrder() >= fe.GetOrder() &&
       (!nodes || nodes->FESpace()->GetMaxElementOrder() == 1))
   {
      const auto &center = IntRules.Get(fe.GetGeomType(), 1);
      const auto det = mesh.GetGeometricFactors(center,
                                                GeometricFactors::DETERMINANTS,
                                                mt)->detJ.Read();
      const real_t scale = constant->constant*Geometry::Volume[fe.GetGeomType()]/nd;
      auto Y = Reshape(b.ReadWrite(), nd, ne);
      mfem::forall(nd, ne, [=] MFEM_HOST_DEVICE (int a, int e)
      {
         if (active[e]) { Y(a,e) += scale*det[e]; }
      });
      return;
   }

   QuadratureSpace qs(mesh, ir);
   CoefficientVector values(coeff, qs, CoefficientStorage::COMPRESSED);
   const bool uniform = values.Size() == 1;
   const auto C = values.Read(), W = ir.GetWeights().Read();
   const auto det = mesh.GetGeometricFactors(ir, GeometricFactors::DETERMINANTS,
                                             mt)->detJ.Read();
   Vector weighted(nq*ne, mt);
   auto F = weighted.Write();
   mfem::forall(nq, ne, [=] MFEM_HOST_DEVICE (int i, int e)
   {
      const int k = i+nq*e;
      F[k] = W[i]*det[k]*C[uniform ? 0 : k];
   });
   if (stroud)
   {
      AssembleBernsteinTetLoad(fe, ir, ne, markers, weighted, b);
      return;
   }

   // A general quadrature rule need not factor. Tabulate its basis once,
   // retaining the prescribed points and the restriction's lexicographic order.
   const auto B = Reshape(fe.GetDofToQuad(ir, DofToQuad::FULL).B.Read(), nq, nd);
   const auto native = fe.GetDofMap().Read();
   auto Y = Reshape(b.ReadWrite(), nd, ne);
   mfem::forall(nd, ne, [=] MFEM_HOST_DEVICE (int a, int e)
   {
      if (!active[e]) { return; }
      real_t sum = 0.0;
      for (int i = 0; i < nq; ++i) { sum += B(i,native[a])*F[i+nq*e]; }
      Y(a,e) += sum;
   });
}

void DomainLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                        const Array<int> &markers, Vector &b)
{
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int qorder = oa * fe.GetOrder() + ob;
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule *ir = IntRule ? IntRule : &IntRules.Get(gtype, qorder);

   if (const auto *tet = dynamic_cast<const H1Pos_TetrahedronElement*>(&fe))
   {
      AssembleTetDomainLF(fes, *tet, *ir, !IntRule, Q, markers, b);
      return;
   }

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
