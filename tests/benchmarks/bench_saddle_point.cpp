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

#include "bench.hpp"
#include "fem/lor/lor_batched.hpp"
#include "general/forall.hpp"
#include <memory>

#ifdef MFEM_USE_BENCHMARK

void FormElementToFace3D(int order, Array<int> &element2face)
{
   const int o = order;
   const int op1 = order + 1;

   const int n = o*o*op1; // number of faces per dimension

   element2face.HostWrite();

   for (int iz = 0; iz < o; ++iz)
   {
      for (int iy = 0; iy < o; ++iy)
      {
         for (int ix = 0; ix < o; ++ix)
         {
            const int ivol = ix + iy*o + iz*o*o;
            element2face[0 + 6*ivol] = -1 - (ix + iy*op1 + iz*o*op1); // x = 0
            element2face[1 + 6*ivol] = ix+1 + iy*op1 + iz*o*op1; // x = 1
            element2face[2 + 6*ivol] = -1 - (ix + iy*o + iz*o*op1 + n); // y = 0
            element2face[3 + 6*ivol] = ix + (iy+1)*o + iz*o*op1 + n; // y = 1
            element2face[4 + 6*ivol] = -1 - (ix + iy*o + iz*o*o + 2*n); // z = 0
            element2face[5 + 6*ivol] = ix + iy*o + (iz+1)*o*o + 2*n; // z = 1
         }
      }
   }
}

void FormDiscreteDivergenceMatrix(FiniteElementSpace &fes_rt,
                                  FiniteElementSpace &fes_l2,
                                  SparseMatrix &D,
                                  Array<int> &element2face)
{
   const Mesh &mesh = *fes_rt.GetMesh();
   const int dim = mesh.Dimension();
   const int order = fes_rt.GetMaxElementOrder();
   MFEM_VERIFY(dim == 3, "");

   const int n_rt = fes_rt.GetNDofs();
   const int n_l2 = fes_l2.GetNDofs();

   if (D.Height() != n_l2 || D.Width() != n_rt)
   {
      D.Clear();
      D.OverrideSize(n_l2, n_rt);
   }

   EnsureCapacity(D.GetMemoryI(), n_l2 + 1);
   // Each row always has 2*dim nonzeros (one for each face of the element)
   const int nnz = n_l2*2*dim;
   auto I = D.WriteI();
   MFEM_FORALL(i, n_l2+1, I[i] = 2*dim*i; );

   const int nel_ho = mesh.GetNE();
   const int nface_per_el = dim*pow(order, dim-1)*(order+1);
   const int nvol_per_el = pow(order, dim);

   // element2face is a mapping of size (2*dim, nvol_per_el) such that with a
   // macro element, subelement i (in lexicographic ordering) has faces (also
   // in lexicographic order) given by the entries (j, i).
   if (element2face.Size() != 2*dim*nvol_per_el)
   {
      element2face.SetSize(2*dim*nvol_per_el);
      FormElementToFace3D(order, element2face);
   }

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const auto *R_rt = dynamic_cast<const ElementRestriction*>(
                         fes_rt.GetElementRestriction(ordering));
   const auto gather_rt = Reshape(R_rt->GatherMap().Read(), nface_per_el, nel_ho);

   const auto e2f = Reshape(element2face.Read(), 2*dim, nvol_per_el);

   // Fill J and data
   EnsureCapacity(D.GetMemoryJ(), nnz);
   EnsureCapacity(D.GetMemoryData(), nnz);

   auto J = D.WriteJ();
   auto V = D.WriteData();

   // Loop over L2 DOFs
   MFEM_FORALL(i, n_l2,
   {
      const int i_loc = i%nvol_per_el;
      const int i_el = i/nvol_per_el;

      for (int k = 0; k < 2*dim; ++k)
      {
         const int sjv_loc = e2f(k, i_loc);
         const int jv_loc = (sjv_loc >= 0) ? sjv_loc : -1 - sjv_loc;
         const int sgn1 = (sjv_loc >= 0) ? 1 : -1;
         const int sj = gather_rt(jv_loc, i_el);
         const int j = (sj >= 0) ? sj : -1 - sj;
         const int sgn2 = (sj >= 0) ? 1 : -1;

         J[k + 2*dim*i] = j;
         V[k + 2*dim*i] = sgn1*sgn2;
      }
   });
}

struct DiscreteDivergenceBenchmark
{
   Mesh mesh;
   RT_FECollection fec_rt;
   L2_FECollection fec_l2;
   FiniteElementSpace fes_rt;
   FiniteElementSpace fes_l2;
   SparseMatrix D;
   Array<int> e2f;
   const int dofs_rt;
   double mdofs;

   DiscreteDivergenceBenchmark(int p_, int N_) :
      mesh(Mesh::MakeCartesian3D(N_, N_ ,N_, Element::HEXAHEDRON)),
      fec_rt(p_ - 1, 3, BasisType::GaussLobatto, BasisType::IntegratedGLL),
      fec_l2(p_ - 1, 3, BasisType::IntegratedGLL),
      fes_rt(&mesh, &fec_rt),
      fes_l2(&mesh, &fec_l2),
      dofs_rt(fes_rt.GetTrueVSize()),
      mdofs(0.0)
   {
      std::cout << "RT DOFs: " << dofs_rt << std::endl;
      FormDiscreteDivergenceMatrix(fes_rt, fes_l2, D, e2f);
      tic_toc.Clear();
   }

   void FormDivergence()
   {
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      FormDiscreteDivergenceMatrix(fes_rt, fes_l2, D, e2f);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdofs += 1e-6 * dofs_rt;
   }

   double Mdofs() const { return mdofs / tic_toc.RealTime(); }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,8,1)
// The different sides of the mesh
// The range LOG2_N_SIDES_X defines a mesh size "k" such that the number of
// elements per side of the mesh is given by "log2(k*x)", where x is given by
// the constant SIDE_X.

#define SIDE_X 0.5
#define LOG2_N_SIDES_X bm::CreateDenseRange(2,18,1)

#define MAX_NDOFS 1e8

int64_t ApproxNDofs(int64_t side, int64_t p)
{
   return 3*side*side*side*p*p*(p+1);
}

bool ProblemTooLarge(int side, int p)
{
   return ApproxNDofs(side, p) > MAX_NDOFS;
}

void DivergenceBenchmark(bm::State &state)
{
   const double log2side = SIDE_X*state.range(0);
   const int side = pow(2, log2side);
   const int p = state.range(1);
   if (ProblemTooLarge(side, p)) { state.SkipWithError("MAX_NDOFS"); }
   else
   {
      std::cout << "Running benchmark " << state.range(0) << "/" << p << std::endl;
      DiscreteDivergenceBenchmark b(p, side);
      while (state.KeepRunning()) { b.FormDivergence(); }
      state.counters["MDof/s"] = bm::Counter(b.Mdofs());
      state.counters["dofs"] = bm::Counter(b.dofs_rt);
      state.counters["p"] = bm::Counter(p);
   }
}

BENCHMARK(DivergenceBenchmark)
   -> ArgsProduct({LOG2_N_SIDES_X,P_ORDERS})
   -> Unit(bm::kMillisecond);


int main(int argc, char *argv[])
{
   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string device_config = "cpu";
   if (bmi::global_context != nullptr)
   {
      const auto device = bmi::global_context->find("device");
      if (device != bmi::global_context->end())
      {
         mfem::out << device->first << " : " << device->second << std::endl;
         device_config = device->second;
      }
   }
   Device device(device_config.c_str());
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }
   bm::RunSpecifiedBenchmarks(&CR);
   return 0;
}

#endif // MFEM_USE_BENCHMARK
