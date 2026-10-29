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

#include "unit_tests.hpp"
#include "mfem.hpp"
#include "general/forall.hpp"

using namespace mfem;

TEST_CASE("CPU diffusion PA beyond fixed quadrature buffers",
          "[DiffusionPA][Parallel]")
{
   if (Device::Allows(Backend::DEVICE_MASK)) { return; }
   for (int dim : {2, 3}) for (bool symmetric : {false, true})
      {
         Mesh mesh = dim == 2 ? Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, true,
                                                      1., 1.)
                     : Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON, 1., 1., 1.);
         mesh.SetCurvature(2);
         mesh.Transform([](const Vector & x, Vector & y)
         {
            y = x;
            real_t bubble = 1.;
            for (int d = 0; d < x.Size(); ++d) { bubble *= x[d] * (1. - x[d]); }
            for (int d = 0; d < x.Size(); ++d) { y[d] += 0.2 * (d + 1) * bubble; }
         });
         H1_FECollection fec(2, dim); FiniteElementSpace fes(&mesh, &fec);
         const auto geom = mesh.GetElementGeometry(0);
         // A modest trial space with a deliberately oversized quadrature rule
         // permits an independent, dense element assembly reference.
         const IntegrationRule &ir = IntRules.Get(geom, 2 * DofQuadLimits::MAX_Q1D + 1);
         DenseMatrix tensor(dim); tensor = 0.;
         for (int i = 0; i < dim; ++i)
         {
            tensor(i, i) = 2. + i;
            for (int j = 0; j < i; ++j)
            { tensor(i, j) = 0.2; tensor(j, i) = symmetric ? 0.2 : 0.4; }
         }
         MatrixConstantCoefficient coefficient(tensor);
         BilinearForm pa(&fes), assembled(&fes);
         pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         auto *pa_integrator = new DiffusionIntegrator(coefficient);
         auto *fa_integrator = new DiffusionIntegrator(coefficient);
         pa_integrator->SetIntRule(&ir); fa_integrator->SetIntRule(&ir);
         pa.AddDomainIntegrator(pa_integrator);
         assembled.AddDomainIntegrator(fa_integrator);
         pa.Assemble(); assembled.Assemble(); assembled.Finalize();
         Vector x(fes.GetVSize()), y(x.Size()), reference(x.Size()); x.Randomize(19);
         pa.Mult(x, y); assembled.Mult(x, reference); y -= reference;
         REQUIRE(y.Norml2() < 1.e-10 * reference.Norml2());
      }
}

TEST_CASE("CPU diffusion PA beyond fixed degree buffers",
          "[DiffusionPA][Parallel]")
{
   if (Device::Allows(Backend::DEVICE_MASK)) { return; }
   const int p = DofQuadLimits::MAX_D1D, n = p + 1;
   H1_SegmentElement segment(p);
   const auto &map1 = segment.GetDofMap();
   DenseMatrix mass(n), stiffness(n), gradient(n, 1); mass = 0.; stiffness = 0.;
   Vector shape(n);
   const auto &ir = IntRules.Get(Geometry::SEGMENT, 2 * p + 1);
   for (int q = 0; q < ir.GetNPoints(); ++q)
   {
      const auto &ip = ir.IntPoint(q); segment.CalcShape(ip, shape);
      segment.CalcDShape(ip, gradient);
      for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
         {
            mass(i, j) += ip.weight * shape[map1[i]] * shape[map1[j]];
            stiffness(i, j) += ip.weight * gradient(map1[i], 0) * gradient(map1[j], 0);
         }
   }
   for (int dim : {2, 3})
   {
      Mesh mesh = dim == 2 ? Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, true,
                                                   1., 1.)
                  : Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON, 1., 1., 1.);
      H1_FECollection fec(p, dim); FiniteElementSpace fes(&mesh, &fec);
      BilinearForm pa(&fes); pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      pa.AddDomainIntegrator(new DiffusionIntegrator); pa.Assemble();
      const auto &map = dynamic_cast<const TensorBasisElement*>(fes.GetFE(
                                                                   0))->GetDofMap();
      Array<int> dofs; fes.GetElementDofs(0, dofs);
      CAPTURE(dim, p);
      const int nd = fes.GetVSize();
      Vector x(nd), y(nd), reference(nd), u(nd), v(nd); x.Randomize(23);
      reference = 0.;
      pa.Mult(x, y);
      // Independent Cartesian stiffness assembled as a sum of Kronecker
      // products of exact one-dimensional mass and stiffness matrices.
      for (int component = 0; component < dim; ++component)
      {
         for (int i = 0; i < nd; ++i) { u[i] = x[dofs[map[i]]]; }
         int stride = 1;
         for (int axis = 0; axis < dim; ++axis)
         {
            const DenseMatrix &a = axis == component ? stiffness : mass;
            for (int i = 0; i < nd; ++i)
            {
               const int row = (i / stride) % n, base = i - row * stride;
               real_t value = 0.;
               for (int j = 0; j < n; ++j) { value += a(row, j) * u[base + j * stride]; }
               v[i] = value;
            }
            u.Swap(v); stride *= n;
         }
         for (int i = 0; i < nd; ++i) { reference[dofs[map[i]]] += u[i]; }
      }
      y -= reference; REQUIRE(y.Norml2() < 1.e-10 * reference.Norml2());
   }
}
