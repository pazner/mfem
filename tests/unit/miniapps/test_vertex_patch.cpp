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
#include "../../../miniapps/solvers/vertex_patch.hpp"
#include <cmath>

using namespace mfem;
using namespace mfem::vertex_patch;
#if defined(MFEM_USE_MPI) && defined(MFEM_USE_LAPACK) && defined(MFEM_USE_METIS) && defined(MFEM_USE_METIS_5)
namespace
{
real_t Dot(const Vector &a, const Vector &b) { return InnerProduct(MPI_COMM_WORLD, a, b); }
Mesh TestMesh(int dim, bool oriented)
{
   Mesh source = dim == 2 ? Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL,
                                                  true, 1., 1.)
                 : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1., 1., 1.);
   if (!oriented) { return source; }
   Mesh mesh(dim, source.GetNV(), source.GetNE(), 0, dim);
   for (int v = 0; v < source.GetNV(); ++v) { mesh.AddVertex(source.GetVertex(v)); }
   Array<int> vertices;
   for (int e = 0; e < source.GetNE(); ++e)
   {
      source.GetElementVertices(e, vertices);
      Array<int> rotated(vertices.Size());
      if (dim == 2)
      {
         for (int i = 0; i < 4; ++i) { rotated[i] = vertices[(i + e) % 4]; }
         mesh.AddQuad(rotated);
      }
      else
      {
         // Proper rotations, including face-axis permutations and reflections.
         const int rotation[8] = {1, 2, 3, 0, 5, 6, 7, 4};
         const int cycle[8] = {0, 3, 7, 4, 1, 2, 6, 5};
         for (int i = 0; i < 8; ++i) { rotated[i] = vertices[i]; }
         for (int k = 0; k < e % 4; ++k)
         { auto old = rotated; for (int i = 0; i < 8; ++i) { rotated[i] = old[rotation[i]]; } }
         if (e % 2) { auto old = rotated; for (int i = 0; i < 8; ++i) { rotated[i] = old[cycle[i]]; } }
         mesh.AddHex(rotated);
      }
   }
   mesh.FinalizeMesh(); return mesh;
}
Array<int> Partition(const Mesh &mesh)
{
   Array<int> part(mesh.GetNE());
   for (int e = 0; e < mesh.GetNE(); ++e) { part[e] = e % Mpi::WorldSize(); }
   return part;
}
}

TEST_CASE("FDM reference basis", "[VertexPatch][Parallel]")
{
   for (int p : {1, 2, 3, 4, 7, 12})
   {
      CAPTURE(p);
      auto b = Basis::Get(p);
      REQUIRE(b == Basis::Get(p));
      DenseMatrix identity(p + 1); Mult(b->S, b->inverse, identity);
      for (int i = 0; i <= p; ++i) for (int j = 0; j <= p; ++j)
         { REQUIRE(identity(i, j) == MFEM_Approx(i == j ? 1. : 0., 1.e-11)); }
      for (int j = 1; j < p; ++j)
      {
         REQUIRE(b->S(0, j) == 0.); REQUIRE(b->S(p, j) == 0.);
         for (int i = 0; i <= p; ++i)
         { REQUIRE(b->S(p - i, j) == MFEM_Approx(b->parity[j]*b->S(i, j), 1.e-12)); }
      }
   }
}

TEST_CASE("FDM sparse auxiliary matrix", "[VertexPatch][Parallel]")
{
   for (int dim : {2, 3}) for (int p : {1, 2, 3, 4})
      {
         CAPTURE(dim, p);
         Mesh mesh = dim == 2 ? Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, true,
                                                      2., 3.)
                     : Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON, 2., 3., 4.);
         H1_FECollection fec(p, dim); FiniteElementSpace fes(&mesh, &fec);
         auto b = Basis::Get(p);
         ConstantCoefficient one(1.);
         auto sparse = AssembleElement(*b, *mesh.GetElementTransformation(0), one);
         DiffusionIntegrator integrator(one);
         DenseMatrix physical;
         integrator.AssembleElementMatrix(*fes.GetFE(0),
                                          *mesh.GetElementTransformation(0), physical);
         const auto *tensor = dynamic_cast<const TensorBasisElement*>(fes.GetFE(0));
         const auto &map = tensor->GetDofMap();
         const int nd = physical.Height(), n = p + 1;
         DenseMatrix s(nd), tmp(nd), transformed(nd);
         for (int i = 0; i < nd; ++i) for (int j = 0; j < nd; ++j)
            {
               int a = i, c = j; real_t value = 1.;
               for (int d = 0; d < dim; ++d) { value *= b->S(a % n, c % n); a /= n; c /= n; }
               s(map[i], j) = value;
            }
         Mult(physical, s, tmp); MultAtB(s, tmp, transformed);
         real_t max_error = 0., scale = transformed.MaxMaxNorm();
         for (int i = 0; i < nd; ++i) for (int j = 0; j < nd; ++j)
            { max_error = std::max(max_error, std::abs(transformed(i, j) - static_cast<const SparseMatrix&>(*sparse)(i, j))); }
         REQUIRE(max_error < 1.e-10 * scale);
      }
}

TEST_CASE("FDM global changes of basis and parallel orientation",
          "[VertexPatch][Parallel]")
{
   for (int dim : {2, 3}) for (bool oriented : {false, true}) for (int p : {2, 3, 4})
         {
            CAPTURE(dim, oriented, p);
            Mesh serial = TestMesh(dim, oriented); auto part = Partition(serial);
            ParMesh mesh(MPI_COMM_WORLD, serial, part);
            H1_FECollection fec(p, dim); ParFiniteElementSpace fes(&mesh, &fec);
            BasisChange change(fes, Basis::Get(p));
            Vector x(fes.GetTrueVSize()), y(x.Size()), sx(x.Size()), sy(x.Size()),
                   z(x.Size());
            x.Randomize(17 + Mpi::WorldRank()); y.Randomize(29 + Mpi::WorldRank());
            change.Mult(x, sx); change.InverseMult(sx, z); z -= x;
            REQUIRE(std::sqrt(Dot(z, z) / Dot(x, x)) < 1.e-11);
            change.MultTranspose(y, sy);
            REQUIRE(Dot(sx, y) == MFEM_Approx(Dot(x, sy), 1.e-10, 1.e-11));
            change.InverseMult(x, sx); change.InverseMultTranspose(y, sy);
            REQUIRE(Dot(sx, y) == MFEM_Approx(Dot(x, sy), 1.e-10, 1.e-11));
            // On Cartesian cells, the assembled auxiliary energy equals the true
            // energy after changing basis, also across rotated shared entities.
            ConstantCoefficient one(1.); ParBilinearForm form(&fes);
            form.AddDomainIntegrator(new DiffusionIntegrator(one)); form.Assemble();
            form.Finalize();
            std::unique_ptr<HypreParMatrix> matrix(form.ParallelAssemble());
            change.Mult(x, sx); matrix->Mult(sx, z); const real_t energy = Dot(sx, z);
            Vector local(fes.GetVSize()); fes.GetProlongationMatrix()->Mult(x, local);
            const int nd = fes.GetFE(0)->GetDof(); real_t auxiliary_energy = 0.;
            Vector el(nd), applied(nd);
            for (int e = 0; e < mesh.GetNE(); ++e)
            {
               auto a = AssembleElement(*Basis::Get(p), *mesh.GetElementTransformation(e),
                                        one);
               for (int i = 0; i < nd; ++i) { el[i] = change.Signs()[e * nd + i] * local[change.ModalMap()[e * nd + i]]; }
               a->Mult(el, applied); auxiliary_energy += el * applied;
            }
            MPI_Allreduce(MPI_IN_PLACE, &auxiliary_energy, 1, MPITypeMap<real_t>::mpi_type,
                          MPI_SUM, MPI_COMM_WORLD);
            REQUIRE(energy == MFEM_Approx(auxiliary_energy, 1.e-9, 1.e-10));
         }
}

TEST_CASE("Vertex-patch preconditioner symmetry and Poisson solves",
          "[VertexPatch][Parallel]")
{
   for (int dim : {2, 3}) for (int p : {1, 2, 4}) for (bool sc : {false, true})
         {
            CAPTURE(dim, p, sc);
            Mesh serial = TestMesh(dim, true); auto part = Partition(serial);
            ParMesh mesh(MPI_COMM_WORLD, serial, part);
            H1_FECollection fec(p, dim); ParFiniteElementSpace fes(&mesh, &fec);
            Array<int> boundary(mesh.bdr_attributes.Max()); boundary = 1;
            Array<int> essential; fes.GetEssentialTrueDofs(boundary, essential);
            ConstantCoefficient one(1.); Options options; options.static_condensation = sc;
            Preconditioner prec(fes, one, boundary, options);
            Vector x(fes.GetTrueVSize()), y(x.Size()), mx(x.Size()), my(x.Size());
            x.Randomize(113 + Mpi::WorldRank()); y.Randomize(211 + Mpi::WorldRank());
            x.SetSubVector(essential, 0.); y.SetSubVector(essential, 0.);
            prec.Mult(x, mx); prec.Mult(y, my);
            REQUIRE(Dot(mx, y) == MFEM_Approx(Dot(x, my), 1.e-9, 1.e-10));
            REQUIRE(Dot(x, mx) > 0.);
            ParBilinearForm form(&fes); form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            form.AddDomainIntegrator(new DiffusionIntegrator(one)); form.Assemble();
            OperatorPtr matrix; form.FormSystemMatrix(essential, matrix);
            Vector rhs(x.Size()), solution(x.Size()); matrix->Mult(x, rhs); solution = 0.;
            CGSolver cg(MPI_COMM_WORLD); cg.SetOperator(*matrix);
            cg.SetPreconditioner(prec);
            cg.SetRelTol(1.e-11); cg.SetMaxIter(200); cg.SetPrintLevel(0);
            cg.Mult(rhs, solution);
            REQUIRE(cg.GetConverged());
            matrix->Mult(solution, my); my -= rhs;
            REQUIRE(std::sqrt(Dot(my, my) / Dot(rhs, rhs)) < 1.e-8);
            solution -= x; REQUIRE(std::sqrt(Dot(solution, solution) / Dot(x, x)) < 1.e-8);
         }
}

TEST_CASE("FDM sparse condensation agrees with dense Schur complements",
          "[VertexPatch][Parallel]")
{
   for (int dim : {2, 3}) for (int p : {2, 3, 4})
      {
         Mesh mesh = TestMesh(dim, false);
         FunctionCoefficient alpha([](const Vector & x) { return 1. + x * x; });
         auto a = AssembleElement(*Basis::Get(p), *mesh.GetElementTransformation(0),
                                  alpha);
         std::vector<int> boundary;
         auto schur = CondenseElement(*a, p, dim, boundary);
         std::vector<int> interior;
         for (int i = 0; i < a->Height(); ++i)
         { if (!std::binary_search(boundary.begin(), boundary.end(), i)) { interior.push_back(i); } }
         DenseMatrix ii(int(interior.size())), ib(int(interior.size()),
                                                  int(boundary.size()));
         const SparseMatrix &mat = *a;
         for (int i = 0; i < int(interior.size()); ++i)
         {
            for (int j = 0; j < int(interior.size()); ++j) { ii(i, j) = mat(interior[i], interior[j]); }
            for (int j = 0; j < int(boundary.size()); ++j) { ib(i, j) = mat(interior[i], boundary[j]); }
         }
         DenseMatrixInverse inverse(ii, true);
         DenseMatrix inverse_ib(ib.Height(), ib.Width()), correction(ib.Width());
         inverse.Mult(ib, inverse_ib); MultAtB(ib, inverse_ib, correction);
         const SparseMatrix &s = *schur;
         for (int i = 0; i < int(boundary.size()); ++i) for (int j = 0;
                                                                j < int(boundary.size()); ++j)
            { REQUIRE(s(i, j) == MFEM_Approx(mat(boundary[i], boundary[j]) - correction(i, j), 1.e-10, 1.e-11)); }
      }
}

TEST_CASE("Vertex patches on curved cells with variable diffusion",
          "[VertexPatch][Parallel]")
{
   for (int dim : {2, 3}) for (bool sc : {false, true})
      {
         Mesh serial = TestMesh(dim, true);
         serial.SetCurvature(2);
         serial.Transform([](const Vector & x, Vector & y)
         {
            y = x; real_t bubble = 1.;
            for (int d = 0; d < x.Size(); ++d) { bubble *= std::sin(std::acos(-1.) * x[d]); }
            for (int d = 0; d < x.Size(); ++d) { y[d] += 0.04 * (d + 1) * bubble; }
         });
         auto part = Partition(serial); ParMesh mesh(MPI_COMM_WORLD, serial, part);
         H1_FECollection fec(4, dim); ParFiniteElementSpace fes(&mesh, &fec);
         Array<int> boundary(mesh.bdr_attributes.Max()); boundary = 1;
         Array<int> essential; fes.GetEssentialTrueDofs(boundary, essential);
         FunctionCoefficient coefficient([](const Vector & x) { return 1. + x * x; });
         Options options; options.static_condensation = sc;
         Preconditioner prec(fes, coefficient, boundary, options);
         ParBilinearForm form(&fes); form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         form.AddDomainIntegrator(new DiffusionIntegrator(coefficient)); form.Assemble();
         OperatorPtr a; form.FormSystemMatrix(essential, a);
         Vector exact(fes.GetTrueVSize()), rhs(exact.Size()), solution(exact.Size()),
                residual(exact.Size());
         exact.Randomize(51 + Mpi::WorldRank()); exact.SetSubVector(essential, 0.);
         a->Mult(exact, rhs); solution = 0.;
         CGSolver cg(MPI_COMM_WORLD); cg.SetOperator(*a); cg.SetPreconditioner(prec);
         cg.SetRelTol(1.e-11); cg.SetMaxIter(200); cg.SetPrintLevel(0);
         cg.Mult(rhs, solution);
         REQUIRE(cg.GetConverged());
         a->Mult(solution, residual); residual -= rhs;
         REQUIRE(std::sqrt(Dot(residual, residual) / Dot(rhs, rhs)) < 1.e-8);
      }
}

TEST_CASE("Condensed single-element Dirichlet problem",
          "[VertexPatch][Parallel]")
{
   // No free interface unknowns: the preconditioner is just the interior solve.
   if (Mpi::WorldSize() != 1) { return; }
   Mesh serial = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON, 1., 1., 1.);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection fec(3, 3); ParFiniteElementSpace fes(&mesh, &fec);
   Array<int> boundary(mesh.bdr_attributes.Max()); boundary = 1;
   ConstantCoefficient one(1.); Preconditioner prec(fes, one, boundary);
   Array<int> essential; fes.GetEssentialTrueDofs(boundary, essential);
   ParBilinearForm form(&fes);
   form.AddDomainIntegrator(new DiffusionIntegrator(one)); form.Assemble();
   HypreParMatrix a; form.FormSystemMatrix(essential, a);
   Vector x(fes.GetTrueVSize()), rhs(x.Size()), y(x.Size());
   x.Randomize(29); x.SetSubVector(essential, 0.); a.Mult(x, rhs);
   prec.Mult(rhs, y); y -= x;
   REQUIRE(y.Norml2() < 1.e-10);
}
#endif
