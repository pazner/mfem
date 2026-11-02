// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE and NOTICE for details.

#include "mfem.hpp"
#include "unit_tests.hpp"
#include "miniapps/solvers/lor_mms.hpp"

using namespace mfem;

TEST_CASE("LOR manufactured load agrees with pointwise evaluation",
          "[LORMMS][GPU]")
{
   const char *file = GENERATE("../../data/star-q3.mesh",
                               "../../data/fichera-q3.mesh",
                               "../../data/star-surf.mesh",
                               "../../data/inline-tri.mesh");
   CAPTURE(file);
   Mesh mesh(file, 1, 1);
   LORScalarLoadCoefficient batched(1.0);
   FunctionCoefficient reference(f(1.0));
   QuadratureSpace qs(&mesh, 14);
   QuadratureFunction actual(qs), expected(qs);
   batched.Project(actual);
   reference.Project(expected);
   actual -= expected;
   REQUIRE(actual.Norml2() < 1.e-12*expected.Norml2());

   // Face rules must retain their own orientation/permutation and geometry.
   FaceQuadratureSpace faces(mesh, 6, FaceType::Boundary);
   QuadratureFunction face_actual(faces), face_expected(faces);
   batched.Project(face_actual);
   reference.Project(face_expected);
   face_actual -= face_expected;
   REQUIRE(face_actual.Norml2() < 1.e-12*std::max(1., face_expected.Norml2()));
}

TEST_CASE("LOR boundary projection preserves the linear system",
          "[LORMMS][GPU]")
{
   const char *file = GENERATE("../../data/star-q3.mesh",
                               "../../data/fichera-q3.mesh",
                               "../../data/star-surf.mesh",
                               "../../data/amr-hex.mesh");
   CAPTURE(file);
   Mesh mesh(file, 1, 1);
   H1_FECollection fec(3, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   FunctionCoefficient exact(u);
   GridFunction full(&fes), boundary(&fes);
   full.ProjectCoefficient(exact);
   boundary = 0.;
   Array<int> marker(mesh.bdr_attributes.Max()), essential;
   marker = 1;
   boundary.ProjectBdrCoefficient(exact, marker);
   fes.GetEssentialTrueDofs(marker, essential);

   BilinearForm a(&fes);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.AddDomainIntegrator(new MassIntegrator);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();
   Vector rhs(fes.GetVSize()), X1, B1, X2, B2;
   rhs = 1.;
   OperatorHandle A1, A2;
   a.FormLinearSystem(essential, full, rhs, A1, X1, B1);
   a.FormLinearSystem(essential, boundary, rhs, A2, X2, B2);
   Vector dx(X1), db(B1);
   dx -= X2; db -= B2;
   REQUIRE(dx.Norml2() < 1.e-12*std::max(1., X1.Norml2()));
   REQUIRE(db.Norml2() < 1.e-12*std::max(1., B1.Norml2()));
}
