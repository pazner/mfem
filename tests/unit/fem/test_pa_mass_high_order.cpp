// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE and NOTICE for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

TEST_CASE("Degree seven mass with twelve quadrature points",
          "[PartialAssembly][MassPA][GPU]")
{
   Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
   mesh.SetCurvature(3);
   H1_FECollection fec(7, 3);
   FiniteElementSpace fes(&mesh, &fec);
   const auto &ir = MassIntegrator::GetRule(*fes.GetFE(0), *fes.GetFE(0),
                                            *mesh.GetElementTransformation(0));
   REQUIRE(ir.GetNPoints() == 12*12*12);

   // Independent tensor-product mass reference on the unit cube.
   H1_SegmentElement segment(7);
   DenseMatrix mass(8); mass = 0.;
   Vector shape(8);
   const auto &rule = IntRules.Get(Geometry::SEGMENT, 22);
   const auto &map1 = segment.GetDofMap();
   for (int q = 0; q < rule.GetNPoints(); ++q)
   {
      const auto &ip = rule.IntPoint(q);
      segment.CalcShape(ip, shape);
      for (int j = 0; j < 8; ++j)
      {
         for (int i = 0; i < 8; ++i)
         { mass(i,j) += ip.weight*shape[map1[i]]*shape[map1[j]]; }
      }
   }
   const auto &map = dynamic_cast<const TensorBasisElement*>(fes.GetFE(
                                                                0))->GetDofMap();
   Array<int> dofs;
   fes.GetElementDofs(0, dofs);
   Vector x(fes.GetVSize()), expected(x.Size()), diagonal(x.Size()),
          actual(x.Size());
   x.Randomize(37); expected = 0.;
   for (int i = 0; i < 512; ++i)
   {
      const int row = dofs[map[i]];
      diagonal[row] = mass(i%8,i%8)*mass(i/8%8,i/8%8)*mass(i/64,i/64);
      for (int j = 0; j < 512; ++j)
      {
         expected[row] += mass(i%8,j%8)*mass(i/8%8,j/8%8)*mass(i/64,
                                                               j/64)*x[dofs[map[j]]];
      }
   }
   BilinearForm a(&fes);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.AddDomainIntegrator(new MassIntegrator);
   a.Assemble();
   a.Mult(x, actual); actual -= expected;
   REQUIRE(actual.Norml2() < 1.e-12*expected.Norml2());
   a.MultTranspose(x, actual); actual -= expected;
   REQUIRE(actual.Norml2() < 1.e-12*expected.Norml2());
   // Application registration must leave the safe diagonal fallback intact.
   a.AssembleDiagonal(actual); actual -= diagonal;
   REQUIRE(actual.Norml2() < 1.e-12*diagonal.Norml2());
}
