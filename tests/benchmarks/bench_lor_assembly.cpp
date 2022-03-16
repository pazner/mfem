// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

#ifdef MFEM_USE_BENCHMARK

#include "fem/lor.hpp"
#include "fem/lor_assembly.hpp"

#define MFEM_DEBUG_COLOR 119
#include "general/debug.hpp"

#include <cassert>
#include <cmath>

constexpr int SEED = 0x100001b3;

struct LORBench
{
   const int p, c, q, n, nx, ny, nz, dim = 3;
   const bool check_x, check_y, check_z, checked;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace mfes, fes_ho;
   Array<int> ess_bdr_ho, ess_dofs_ho;
   LORDiscretization lor_disc;
   IntegrationRules irs;
   const IntegrationRule &ir_el;
   FiniteElementSpace &fes_lo;
   Array<int> ess_bdr_lo, ess_dofs_lo, ess_tdofs_empty;
   BilinearForm a_legacy, a_full;
   OperatorHandle A_batched, A_deviced;
   SparseMatrix *A_full;
   GridFunction x;
   const int dofs;
   double mdof;

   LORBench(int p, int side):
      p(p),
      c(side),
      q(2*p + 2),
      n((assert(c>=p),c/p)),
      nx(n + (p*(n+1)*p*n*p*n < c*c*c ?1:0)),
      ny(n + (p*(n+1)*p*(n+1)*p*n < c*c*c ?1:0)),
      nz(n),
      check_x(p*nx * p*ny * p*nz <= c*c*c),
      check_y(p*(nx+1) * p*(ny+1) * p*nz > c*c*c),
      check_z(p*(nx+1) * p*(ny+1) * p*(nz+1) > c*c*c),
      checked((assert(check_x && check_y && check_z), true)),
      mesh(Mesh::MakeCartesian3D(nx,ny,nz, Element::HEXAHEDRON)),
      fec(p, dim, BasisType::GaussLobatto),
      mfes(&mesh, &fec, dim),
      fes_ho(&mesh, &fec),
      ess_bdr_ho(mesh.bdr_attributes.Max()),
      lor_disc(fes_ho, BasisType::GaussLobatto),
      irs(0, Quadrature1D::GaussLobatto),
      ir_el(irs.Get(Geometry::Type::CUBE, 1)),
      fes_lo(lor_disc.GetFESpace()),
      ess_bdr_lo(fes_lo.GetMesh()->bdr_attributes.Max()),
      a_legacy(&fes_lo),
      a_full(&fes_lo),
      A_deviced(),
      A_full(nullptr),
      x(&mfes),
      dofs(fes_ho.GetVSize()),
      mdof(0.0)
   {
      dbg("p:%d side:%d dofs:%d/%d",p,side,dofs, fes_lo.GetVSize());
      a_legacy.AddDomainIntegrator(new DiffusionIntegrator(&ir_el));
      a_legacy.SetAssemblyLevel(AssemblyLevel::LEGACY);

      a_full.AddDomainIntegrator(new DiffusionIntegrator(&ir_el));
      a_full.SetAssemblyLevel(AssemblyLevel::FULL);

      SetupRandomMesh();
      // Make sure that SetCurvature is called on the LOR mesh
      fes_lo.GetMesh()->EnsureNodes();

      ess_bdr_ho = 1;
      fes_ho.GetEssentialTrueDofs(ess_bdr_ho, ess_dofs_ho);

      ess_bdr_lo = 1;
      fes_lo.GetEssentialTrueDofs(ess_bdr_lo, ess_dofs_lo);

      tic_toc.Clear();
   }

   void SetupRandomMesh() noexcept
   {
      mesh.SetNodalFESpace(&mfes);
      mesh.SetNodalGridFunction(&x);
      const double jitter = 1./(M_PI*M_PI);
      const double h0 = mesh.GetElementSize(0);
      GridFunction rdm(&mfes);
      rdm.Randomize(SEED);
      rdm -= 0.5; // Shift to random values in [-0.5,0.5]
      rdm *= jitter * h0; // Scale the random values to be of same order
      x -= rdm;
   }

   void SanityChecks()
   {
      dbg();
      constexpr double EPS = 1e-15;

      Vector x(dofs), y(dofs);
      x.Randomize(SEED);
      y.Randomize(SEED);

      OperatorHandle A_legacy, A_full, A_deviced;

      BilinearForm a_legacy(&fes_lo);
      a_legacy.AddDomainIntegrator(new DiffusionIntegrator(&ir_el));
      a_legacy.SetAssemblyLevel(AssemblyLevel::LEGACY);
      MFEM_DEVICE_SYNC;
      tic();
      a_legacy.Assemble();
      MFEM_DEVICE_SYNC;
      dbg(" Legacy time = %f",toc());
      a_legacy.FormSystemMatrix(ess_dofs_lo, A_legacy);
      a_legacy.Finalize();
      A_legacy.As<SparseMatrix>()->HostReadWriteI();
      A_legacy.As<SparseMatrix>()->HostReadWriteJ();
      A_legacy.As<SparseMatrix>()->HostReadWriteData();
      const double dot_legacy = A_legacy.As<SparseMatrix>()->InnerProduct(x,y);

      MFEM_DEVICE_SYNC;
      tic();
      a_full.Assemble();
      MFEM_DEVICE_SYNC;
      dbg("   Full time = %f",toc());
      a_full.FormSystemMatrix(ess_dofs_lo, A_full); /// BC NOT DONE !!!
      constexpr bool still_have_to_remove_the_bc = true;
      if (still_have_to_remove_the_bc)
      {
         a_full.EliminateVDofs(ess_dofs_lo, Operator::DIAG_KEEP);
         a_full.Finalize();
      }
      a_full.SpMat().HostReadWriteI();
      a_full.SpMat().HostReadWriteJ();
      a_full.SpMat().HostReadWriteData();
      const double dot_full = a_full.SpMat().InnerProduct(x,y);
      MFEM_VERIFY(almost_equal(dot_legacy, dot_full), "dot_full error!");
      a_full.SpMat().Add(-1.0, *A_legacy.As<SparseMatrix>());
      const double max_norm_full = a_full.SpMat().MaxNorm();
      MFEM_VERIFY(max_norm_full < EPS, "max_norm_full error!");

      MFEM_DEVICE_SYNC;
      tic();
      AssembleBatchedLOR(lor_disc, a_legacy, fes_ho, ess_dofs_lo, A_deviced);
      MFEM_DEVICE_SYNC;
      dbg("Deviced time = %f",toc());
      A_deviced.As<SparseMatrix>()->HostReadWriteI();
      A_deviced.As<SparseMatrix>()->HostReadWriteJ();
      A_deviced.As<SparseMatrix>()->HostReadWriteData();
      const double dot_device = A_deviced.As<SparseMatrix>()->InnerProduct(x,y);
      MFEM_VERIFY(almost_equal(dot_legacy, dot_device), "dot_device error!");
      A_deviced.As<SparseMatrix>()->Add(-1.0, *A_legacy.As<SparseMatrix>());
      const double max_norm_deviced = A_deviced.As<SparseMatrix>()->MaxNorm();
      MFEM_VERIFY(max_norm_deviced < EPS, "max_norm_deviced");
   }

   void GLVis(Mesh &mesh, GridFunction *x = nullptr)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      if (!x) { sol_sock << "mesh\n" << mesh; }
      else { sol_sock << "solution\n" << mesh << *x ;}
      sol_sock << std::flush;
   }

   void Test()
   {
      MFEM_DEVICE_SYNC;
      tic();
      AssembleBatchedLOR(lor_disc, a_legacy, fes_ho, ess_dofs_lo, A_deviced);
      MFEM_DEVICE_SYNC;
      dbg(" Deviced time = %f",toc());
      A_deviced.Clear(); // forcing initialization phase
      dbg("Exiting!");
      std::exit(0);
   }

   void Dump()
   {
      OperatorHandle A_legacy;

      MFEM_DEVICE_SYNC;
      a_legacy.Assemble();
      MFEM_DEVICE_SYNC;
      a_legacy.FormSystemMatrix(ess_dofs_lo, A_legacy);
      a_legacy.Finalize();
      A_legacy.As<SparseMatrix>()->HostReadWriteI();
      A_legacy.As<SparseMatrix>()->HostReadWriteJ();
      A_legacy.As<SparseMatrix>()->HostReadWriteData();

      dbg("Saving 'A.mtx' file");
      {
         std::ofstream mtx_file("A.mtx");
         A_legacy.As<SparseMatrix>()->PrintMM(mtx_file);
      }

      dbg("fes_lo.GetVSize: %d", fes_lo.GetVSize());
      Mesh *mesh_lo =fes_lo.GetMesh();
      GridFunction ids_lo(&fes_lo);
      Array<int> dofs(fes_lo.GetVSize());
      fes_lo.GetVDofs(0,dofs);
      assert(fes_lo.GetVDim()==1);
      for (int i=0; i<fes_lo.GetVSize(); i++) { ids_lo(i) = dofs[i]; }
      GLVis(*mesh_lo, &ids_lo);
      dbg("LO GLVis done!");

      dbg("fes_ho.GetVSize: %d", fes_ho.GetVSize());
      GridFunction ids(&fes_ho);
      Array<int> vdofs(fes_ho.GetVSize());
      fes_ho.GetVDofs(0, vdofs);
      assert(fes_ho.GetVDim()==1);
      for (int i=0; i<fes_ho.GetVSize(); i++) { ids(i) = vdofs[i]; }
      GLVis(mesh, &ids);
      dbg("HO GLVis done!");

      dbg("Exiting!");
      std::exit(0);
   }

   void KerLegacy()
   {
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      a_legacy.Assemble();
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void KerFull()
   {
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      a_full.Assemble();
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void KerBatched()
   {
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      AssembleBatchedLOR(lor_disc, a_legacy, fes_ho, ess_dofs_lo, A_deviced);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void AllFull()
   {
      LORDiscretization lor_disc(fes_ho, BasisType::GaussLobatto);
      FiniteElementSpace fes_lo(lor_disc.GetFESpace());
      BilinearForm bf_full(&fes_lo);
      bf_full.AddDomainIntegrator(new DiffusionIntegrator(&ir_el));
      bf_full.SetAssemblyLevel(AssemblyLevel::FULL);
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      bf_full.Assemble();
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void AllBatched()
   {
      LORDiscretization lor_disc(fes_ho, BasisType::GaussLobatto);
      FiniteElementSpace fes_lo(lor_disc.GetFESpace());
      BilinearForm a_legacy(&fes_lo);
      a_legacy.AddDomainIntegrator(new DiffusionIntegrator(&ir_el));
      a_legacy.SetAssemblyLevel(AssemblyLevel::LEGACY);
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      OperatorHandle A_deviced;
      AssembleBatchedLOR(lor_disc, a_legacy, fes_ho, ess_tdofs_empty, A_deviced);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   double Mdofs() const { return mdof / tic_toc.RealTime(); }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,8,1)

// The different sides of the mesh
#define N_SIDES bm::CreateDenseRange(6,120,6)
#define MAX_NDOFS 2*1024*1024

/// Kernels definitions and registrations
#define Benchmark(Name)\
static void Name(bm::State &state){\
   const int p = state.range(0);\
   const int side = state.range(1);\
   LORBench lor(p, side);\
   if (lor.dofs > MAX_NDOFS) { state.SkipWithError("MAX_NDOFS"); }\
   while (state.KeepRunning()) { lor.Name(); }\
   bm::Counter::Flags flags = bm::Counter::kIsIterationInvariantRate;\
   state.counters["Ker_(MDof/s)"] = bm::Counter(1e-6*lor.dofs, flags);\
   state.counters["All_(MDof/s)"] = bm::Counter(lor.Mdofs());\
   state.counters["dofs"] = bm::Counter(lor.dofs);\
   state.counters["p"] = bm::Counter(p);\
}\
BENCHMARK(Name)\
            -> ArgsProduct({P_ORDERS,N_SIDES})\
            -> Unit(bm::kMillisecond)\
            -> Iterations(10);

Benchmark(SanityChecks)

Benchmark(KerLegacy)
Benchmark(KerFull)
Benchmark(KerBatched)

Benchmark(AllFull)
Benchmark(AllBatched)

Benchmark(Dump)
Benchmark(Test)

/**
 * @brief main entry point
 * --benchmark_filter=Batched/4/16
 * --benchmark_filter=\(Batched\|Deviced\|Full\)/4/16
 * --benchmark_context=device=cuda
 */
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
