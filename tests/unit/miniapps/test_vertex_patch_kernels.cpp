// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE and NOTICE for details.
#include "unit_tests.hpp"
#include "mfem.hpp"
#include "miniapps/solvers/vertex_patch_solver.hpp"
#include "linalg/kernels.hpp"
#include <array>
#include <map>
#include <tuple>
#include "general/forall.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>
#ifdef MFEM_USE_CUDA
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#endif

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_LAPACK) && defined(MFEM_USE_METIS_5)
namespace mfem
{
namespace vertex_patch
{
namespace
{
#include "miniapps/solvers/vertex_patch_kernels.hpp"
int Power(int n, int dim)
{ int result = 1; for (int d = 0; d < dim; ++d) { result *= n; } return result; }
bool Interior(int i, int n, int dim)
{
   for (int d = 0; d < dim; ++d, i /= n)
   { if (i%n == 0 || i%n == n-1) { return false; } }
   return true;
}
}
#include "miniapps/solvers/vertex_patch_assembly.hpp"
}
}
using namespace mfem;
using namespace mfem::vertex_patch;

TEST_CASE("Batched patch factors agree with prescribed-pattern CPU ICC",
          "[VertexPatch][Parallel]")
{
   for (real_t shift : {0., 0.15})
   {
      PatchFactors batch;
      Vector input(31), expected(31), actual(31);
      input.Randomize(73); expected = 0.;
      Array<int> order(10);
      for (int p = 0; p < 10; ++p)
      {
         const int n = 5+3*(p%5);
         SparseMatrix matrix(n);
         for (int i = 0; i < n; ++i)
         {
            matrix.Add(i,i,5.+0.1*i+0.2*p);
            if (i+1 < n) { matrix.Add(i,i+1,-0.5); matrix.Add(i+1,i,-0.5); }
            if (i+3 < n) { matrix.Add(i,i+3,-0.2); matrix.Add(i+3,i,-0.2); }
            if (i+2 < n) { matrix.Add(i,i+2,0.); matrix.Add(i+2,i,0.); }
         }
         matrix.Finalize(0); matrix.SortColumnIndices();
         matrix.HostReadI(); matrix.HostReadJ(); matrix.HostReadData();
         Array<int> permutation(n), indices(n);
         for (int i = 0; i < n; ++i) { permutation[i] = n-1-i; indices[i] = (i+2*p)%31; }
         IncompleteCholeskySolver reference;
         REQUIRE(reference.Factor(matrix,matrix,permutation,shift) == -1);
         Vector rhs(n), solution(n);
         for (int i = 0; i < n; ++i) { rhs[i] = input[indices[i]]; }
         reference.Mult(rhs,solution);
         for (int i = 0; i < n; ++i) { expected[indices[i]] += solution[i]; }
         order[p] = batch.Add(matrix,permutation,indices);
      }
      REQUIRE(batch.NumPatterns() == 5);
      const int nonzeros = batch.Nonzeros();
      batch.Factor(shift); batch.Prepare(order,3);
      REQUIRE(batch.Nonzeros() == nonzeros);
      input.UseDevice(true); actual.UseDevice(true); actual = 0.;
      // Identical graphs with different values/maps must share only symbolic data.
      batch.Apply(input,actual,order,0,3); batch.Apply(input,actual,order,3,10);
      actual -= expected;
      REQUIRE(actual.Norml2() < 1.e-12*expected.Norml2());
      actual = 0.;
      batch.ApplyLevels(input,actual,false); batch.ApplyLevels(input,actual,true);
      actual -= expected;
      REQUIRE(actual.Norml2() < 1.e-12*expected.Norml2());
      actual = 0.;
      batch.ApplyBlocks<0>(input,actual,order,0,10);
      actual -= expected;
      REQUIRE(actual.Norml2() < 1.e-12*expected.Norml2());

   }
}
TEST_CASE("Batched auxiliary assembly agrees with element reference",
          "[VertexPatch][Parallel]")
{
   for (int dim : {2,3}) for (int p : {2,4}) for (bool condensed : {false,true})
         {
            CAPTURE(dim,p,condensed);
            Mesh serial = dim == 2 ? Mesh::MakeCartesian2D(2,2,Element::QUADRILATERAL,true)
                          : Mesh::MakeCartesian3D(2,2,2,Element::HEXAHEDRON);
            serial.SetCurvature(2);
            serial.Transform([](const Vector &x, Vector &y)
            {
               y = x; real_t bubble = 1.;
               for (int d = 0; d < x.Size(); ++d) { bubble *= x[d]*(1.-x[d]); }
               for (int d = 0; d < x.Size(); ++d) { y[d] += 0.15*(d+1)*bubble; }
            });
            ParMesh mesh(MPI_COMM_WORLD,serial);
            FunctionCoefficient coefficient([](const Vector &x) { return 1.+x*x; });
            auto basis = Basis::Get(p);
            ElementBatch batch(*basis,dim,condensed);
            batch.Assemble(mesh,coefficient,condensed);
            const auto *values = batch.active_values.HostRead();
            const int *I = batch.pattern->HostReadI(), *J = batch.pattern->HostReadJ();
            real_t error = 0., scale = 0.;
            for (int e = 0; e < mesh.GetNE(); ++e)
            {
               auto reference = AssembleElement(*basis,*mesh.GetElementTransformation(e),
                                                coefficient);
               if (condensed)
               {
                  std::vector<int> boundary;
                  reference = CondenseElement(*reference,p,dim,boundary);
               }
               std::map<std::pair<int,int>,real_t> difference;
               const int *ri = reference->HostReadI(), *rj = reference->HostReadJ();
               const real_t *rv = reference->HostReadData();
               for (int i = 0; i < reference->Height(); ++i)
               {
                  for (int k = ri[i]; k < ri[i+1]; ++k)
                  { difference[ {i,rj[k]}] = -rv[k]; scale = std::max(scale,std::abs(rv[k])); }
                  for (int k = I[i]; k < I[i+1]; ++k)
                  { difference[ {i,J[k]}] += values[e*batch.active_nz+k]; }
               }
               for (const auto &entry : difference) { error = std::max(error,std::abs(entry.second)); }
            }
            REQUIRE(error < 2.e-11*scale);
         }
}
TEST_CASE("Reduced coarse transfer preserves linear fields and its adjoint",
          "[VertexPatch][Parallel]")
{
   for (int dim : {2,3}) for (int p : {2,7})
      {
         CAPTURE(dim,p);
         Mesh serial = dim == 2 ? Mesh::MakeCartesian2D(2,2,Element::QUADRILATERAL,true)
                       : Mesh::MakeCartesian3D(2,2,2,Element::HEXAHEDRON);
         ParMesh mesh(MPI_COMM_WORLD,serial);
         H1_FECollection low_collection(1,dim), high_collection(p,dim);
         ParFiniteElementSpace coarse(&mesh,&low_collection), fine(&mesh,
                                                                   &high_collection);
         FunctionCoefficient linear([](const Vector &x)
         { real_t v = 1.; for (int d = 0; d < x.Size(); ++d) { v += (d+1)*x[d]; } return v; });
         ParGridFunction low(&coarse), high(&fine);
         low.ProjectCoefficient(linear); high.ProjectCoefficient(linear);
         Vector x(coarse.GetTrueVSize()), expected(fine.GetTrueVSize()),
                y(expected.Size());
         const real_t *lo = low.HostRead(), *hi = high.HostRead();
         for (int i = 0; i < coarse.GetVSize(); ++i)
         { const int j = coarse.GetLocalTDofNumber(i); if (j >= 0) { x[j] = lo[i]; } }
         for (int i = 0; i < fine.GetVSize(); ++i)
         { const int j = fine.GetLocalTDofNumber(i); if (j >= 0) { expected[j] = hi[i]; } }
         LinearCoarseTransfer transfer(coarse,fine);
         transfer.Mult(x,y); y -= expected;
         REQUIRE(y.Norml2() < 1.e-11*expected.Norml2());
         x.Randomize(31+Mpi::WorldRank()); expected.Randomize(57+Mpi::WorldRank());
         Vector transpose(x.Size());
         transfer.Mult(x,y); transfer.MultTranspose(expected,transpose);
         REQUIRE(InnerProduct(MPI_COMM_WORLD,y,expected) ==
                 MFEM_Approx(InnerProduct(MPI_COMM_WORLD,x,transpose),1.e-11,1.e-11));
      }
}
#endif
