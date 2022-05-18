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
#include "kershaw.hpp"
#include <memory>

#ifdef MFEM_USE_BENCHMARK

Mesh CreateKershawMesh(int N, double eps)
{
   Mesh mesh = Mesh::MakeCartesian3D(N, N, N, Element::HEXAHEDRON);
   KershawTransformation kt(mesh.Dimension(), eps, eps);
   mesh.Transform(kt);
   return mesh;
}

struct RTMassBenchmark
{
   const int p;
   const int N;
   const int dim = 3;
   Mesh mesh;
   RT_FECollection fec;
   FiniteElementSpace fes;
   const int n;

   VectorFEMassIntegrator *integ;
   BilinearForm m;

   Vector X, B;

   const int dofs;
   double mdofs;

   RTMassBenchmark(int p_, int N_, double eps_) :
      p(p_),
      N(N_),
      mesh(CreateKershawMesh(N,eps_)),
      fec(p, dim),
      fes(&mesh, &fec),
      n(fes.GetTrueVSize()),
      integ(new VectorFEMassIntegrator),
      m(&fes),
      X(n),
      B(n),
      dofs(n),
      mdofs(0.0)
   {
      m.AddDomainIntegrator(integ);
      m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      m.Assemble();
      X.Randomize(1);
      tic_toc.Clear();
   }

   void Mult()
   {
      B = 0.0;
      MFEM_DEVICE_SYNC;
      tic_toc.Start();
      m.Mult(X, B);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdofs += 1e-6 * dofs;
   }

   void Fallback()
   {
      integ->fallback = true;
      Mult();
   }

   void Smem()
   {
      integ->fallback = false;
      Mult();
   }

   double Mdofs() const { return mdofs / tic_toc.RealTime(); }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,5,1)
// The different sides of the mesh
#define N_SIDES bm::CreateDenseRange(2,30,2)

#define MAX_NDOFS 2*1024*1024

/// Kernels definitions and registrations
#define Benchmark(op_name, suffix, eps)\
static void op_name##_##suffix(bm::State &state){\
   const int side = state.range(0);\
   const int p = state.range(1);\
   RTMassBenchmark mb(p, side, eps);\
   if (mb.dofs > MAX_NDOFS) { state.SkipWithError("MAX_NDOFS"); }\
   while (state.KeepRunning()) { mb.op_name(); }\
   state.counters["MDof/s"] = bm::Counter(mb.Mdofs());\
   state.counters["dofs"] = bm::Counter(mb.dofs);\
   state.counters["p"] = bm::Counter(p);\
}\
BENCHMARK(op_name##_##suffix)\
            -> ArgsProduct({N_SIDES,P_ORDERS})\
            -> Unit(bm::kMillisecond);

Benchmark(Fallback, 1_0, 1.0)
Benchmark(Smem, 1_0, 1.0)

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
