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


// MPI H1 Poisson with sparse FDM vertex patches (Brubeck--Farrell,
// arXiv:2211.14284, Sections 2.2, 3, 4.4--4.5).
// Build: make vertex_patch
// Runs: mpirun -np 4 vertex_patch -m ../../data/beam-hex.mesh -o 4
//       mpirun -np 4 vertex_patch -m ../../data/star.mesh -o 7 -no-sc
//
// The physical operator uses partial assembly. Static condensation changes
// only the preconditioner. No high-order dense element matrices are formed.
#include "vertex_patch_solver.hpp"
#include "../../general/forall.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#ifdef MFEM_USE_CUDA
#include <cuda_profiler_api.h>
#endif

using namespace mfem;

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_LAPACK) && defined(MFEM_USE_METIS) && defined(MFEM_USE_METIS_5)
// Device projection keeps coefficient evaluation out of host quadrature loops.
class DriverCoefficient : public Coefficient
{
   int kind;
   bool load, manufactured;
public:
   DriverCoefficient(int k, bool rhs = false, bool ms = false)
      : kind(k), load(rhs), manufactured(ms) { }
   MFEM_HOST_DEVICE static real_t Value(const real_t *x, int dim, int kind,
                                        bool load, bool manufactured)
   {
      real_t alpha = 1.;
      if (kind) for (int d = 0; d < dim; ++d) { alpha += x[d]*x[d]; }
      if (!load) { return alpha; }
      if (!manufactured) { return 1.; }
      const real_t pi = 3.14159265358979323846;
      real_t u = 1., gradient = 0.;
      for (int d = 0; d < dim; ++d) { u *= sin(pi*x[d]); }
      if (kind) for (int d = 0; d < dim; ++d)
         {
            real_t du = pi*cos(pi*x[d]);
            for (int j = 0; j < dim; ++j) if (j != d) { du *= sin(pi*x[j]); }
            gradient += 2.*x[d]*du;
         }
      return dim*pi*pi*alpha*u-gradient;
   }
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x; T.Transform(ip,x);
      return Value(x.GetData(),x.Size(),kind,load,manufactured);
   }
   void Project(QuadratureFunction &qf) override
   {
      auto &qs = *qf.GetSpace(); auto &mesh = *qs.GetMesh();
      if (!mesh.GetNE()) { return; }
      const auto &ir = qs.GetIntRule(0); const int nq = ir.GetNPoints();
      const int dim = mesh.Dimension(), k = kind;
      const bool rhs = load, ms = manufactured;
      const auto *geom = mesh.GetGeometricFactors(ir,GeometricFactors::COORDINATES);
      const auto X = Reshape(geom->X.Read(),nq,dim,mesh.GetNE());
      real_t *V = qf.Write();
      mfem::forall(qf.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         real_t x[3];
         for (int d = 0; d < dim; ++d) { x[d] = X(i%nq,d,i/nq); }
         V[i] = Value(x,dim,k,rhs,ms);
      });
   }
};

// Standard Gauss quadrature is exact for constant diffusion on affine tensor
// cells. Preserve the higher quadrature on curved/non-affine cells and for
// variable coefficients. Test before device geometry creates mesh nodes.
bool AffineTensorMesh(ParMesh &mesh)
{
   int affine = mesh.GetNodes() ? 0 : 1;
   const int dim = mesh.Dimension(); Array<int> vertices;
   for (int e = 0; affine && e < mesh.GetNE(); ++e)
   {
      mesh.GetElementVertices(e,vertices);
      const auto &reference = *Geometries.GetVertices(mesh.GetElementGeometry(e));
      const real_t *corner[8] = {};
      for (int j = 0; j < vertices.Size(); ++j)
      {
         const auto &ip = reference.IntPoint(j);
         corner[int(ip.x)+2*int(ip.y)+(dim == 3 ? 4*int(ip.z) : 0)] = mesh.GetVertex(
                                                                         vertices[j]);
      }
      real_t scale = 0.;
      for (int v = 0; v < (1<<dim); ++v) for (int d = 0; d < dim; ++d)
         { scale = std::max(scale,std::abs(corner[v][d]-corner[0][d])); }
      const real_t tolerance = 64*std::numeric_limits<real_t>::epsilon()*scale;
      for (int v = 0; v < (1<<dim); ++v) for (int d = 0; d < dim; ++d)
         {
            real_t value = 0.;
            for (int a = 0; a < dim; ++a)
            { if ((v>>a)&1) { value += corner[1<<a][d]-corner[0][d]; } }
            if (std::abs(value-(corner[v][d]-corner[0][d])) > tolerance) { affine = 0; }
         }
   }
   MPI_Allreduce(MPI_IN_PLACE,&affine,1,MPI_INT,MPI_MIN,mesh.GetComm());
   return affine;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   const int rank = Mpi::WorldRank();
   const char *mesh_file = "../../data/beam-hex.mesh";
   const char *device_config = "cpu";
   int device_id = -1;
   bool gpu_aware_mpi = false, profile = false, overintegrate = false;
   int order = 3, serial_refinements = 0, parallel_refinements = 0,
       max_iterations = 500;
   int coefficient_kind = 0;
   real_t tolerance = 1.e-8;
   bool manufactured = false;
   Array<int> attributes;
   vertex_patch::Options options;
   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device", "MFEM device backend.");
   args.AddOption(&device_id, "-did", "--device-id",
                  "GPU index; -1 selects by node-local MPI rank.");
   args.AddOption(&gpu_aware_mpi, "-gmpi", "--gpu-aware-mpi", "-no-gmpi",
                  "--no-gpu-aware-mpi", "Use device buffers in GPU-aware MPI.");
   args.AddOption(&overintegrate,"-oq","--overintegrate","-no-oq",
                  "--no-overintegrate",
                  "Always use auxiliary GLL quadrature for physical diffusion.");
   args.AddOption(&profile, "-profile", "--profile", "-no-profile", "--no-profile",
                  "Bracket the warmed application/solve with CUDA profiler markers.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Quadrilateral or hexahedral mesh.");
   args.AddOption(&order, "-o", "--order", "Uniform polynomial degree.");
   args.AddOption(&serial_refinements, "-rs", "--serial-refinements",
                  "Serial refinements.");
   args.AddOption(&parallel_refinements, "-rp", "--parallel-refinements",
                  "Parallel refinements.");
   args.AddOption(&attributes, "-eb", "--essential-boundary",
                  "Essential attributes; empty means all.");
   args.AddOption(&coefficient_kind, "-c", "--coefficient",
                  "Diffusion: 0 = one, 1 = 1+sum(x_i^2).");
   args.AddOption(&manufactured, "-ms", "--manufactured", "-no-ms",
                  "--no-manufactured",
                  "Use u=product(sin(pi*x_i)) and its manufactured load on the unit box.");
   args.AddOption(&options.static_condensation, "-sc", "--static-condensation",
                  "-no-sc", "--no-static-condensation", "Condense the auxiliary operator.");
   args.AddOption(&options.damping, "-damp", "--damping",
                  "Patch damping; zero estimates it.");
   args.AddOption(&options.diagonal_shift, "-shift", "--diagonal-shift",
                  "Relative ICC diagonal shift.");
   args.AddOption(&tolerance, "-rtol", "--relative-tolerance",
                  "True relative residual tolerance.");
   args.AddOption(&max_iterations, "-maxit", "--max-iterations",
                  "Maximum CG iterations.");
   args.Parse();
   if (!args.Good()) { if (!rank) { args.PrintUsage(std::cout); } return 1; }
   if (device_id < 0)
   {
      MPI_Comm local; int local_rank;
      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                          MPI_INFO_NULL, &local);
      MPI_Comm_rank(local, &local_rank); MPI_Comm_free(&local);
      int count = 0;
#ifdef MFEM_USE_CUDA
      if (std::string(device_config).find("cuda") != std::string::npos ||
          std::string(device_config).find("gpu") != std::string::npos)
      { MFEM_GPU_CHECK(cudaGetDeviceCount(&count)); }
#endif
      device_id = count ? local_rank % count : 0;
   }
   Device device(device_config, device_id);
   Device::SetGPUAwareMPI(gpu_aware_mpi); Hypre::Init();
   if (!rank) { device.Print(); }
   if (!rank) { args.PrintOptions(std::cout); }
   MFEM_VERIFY(order >= 1 && serial_refinements >= 0 &&
               parallel_refinements >= 0 &&
               max_iterations > 0 && tolerance > 0. && tolerance < 1. &&
               coefficient_kind >= 0 && coefficient_kind <= 1, "Invalid driver options");
   Mesh serial(mesh_file, 1, 1);
   for (int l = 0; l < serial_refinements; ++l) { serial.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial); serial.Clear();
   for (int l = 0; l < parallel_refinements; ++l) { mesh.UniformRefinement(); }
   const int dim = mesh.Dimension();
   int invalid_geometry = mesh.Nonconforming() || (dim != 2 && dim != 3) ||
                          mesh.SpaceDimension() != dim;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      if ((dim == 2 && mesh.GetElementGeometry(e) != Geometry::SQUARE) ||
          (dim == 3 && mesh.GetElementGeometry(e) != Geometry::CUBE)) { invalid_geometry = 1; }
   }
   MPI_Allreduce(MPI_IN_PLACE, &invalid_geometry, 1, MPI_INT, MPI_MAX,
                 MPI_COMM_WORLD);
   MFEM_VERIFY(!invalid_geometry,
               "The driver requires conforming quadrilateral or hexahedral volume meshes");
   const bool standard_quadrature = !overintegrate && coefficient_kind == 0 &&
                                    AffineTensorMesh(mesh);
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&mesh, &fec);
   Array<int> ess_bdr(mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0);
   ess_bdr = attributes.Size() ? 0 : 1;
   for (int attr : attributes)
   { MFEM_VERIFY(attr > 0 && attr <= ess_bdr.Size(), "Invalid essential attribute"); ess_bdr[attr - 1] = 1; }
   MFEM_VERIFY(!manufactured || attributes.Size() == 0,
               "The manufactured driver case uses all-Dirichlet unit-box boundaries");
   DriverCoefficient diffusion(coefficient_kind);
   const real_t pi = std::acos(-1.);
   FunctionCoefficient exact([pi](const Vector & x)
   { real_t u = 1.; for (int i = 0; i < x.Size(); ++i) { u *= std::sin(pi*x[i]); } return u; });
   DriverCoefficient load(coefficient_kind, true, manufactured);
   Array<int> essential; fes.GetEssentialTrueDofs(ess_bdr, essential);
   ParGridFunction x(&fes); x = 0.;
   ParLinearForm b(&fes); b.AddDomainIntegrator(new DomainLFIntegrator(load));
   b.UseFastAssembly(true); b.Assemble();
   ParBilinearForm a(&fes); a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   IntegrationRules rules(0, Quadrature1D::GaussLobatto);
   auto *integrator = new DiffusionIntegrator(diffusion);
   if (standard_quadrature)
   { integrator->SetIntRule(&IntRules.Get(TensorBasisElement::GetTensorProductGeometry(dim),2*order)); }
   else
   { integrator->SetIntRule(&rules.Get(TensorBasisElement::GetTensorProductGeometry(dim), 3 * order)); }
   if (!rank) { std::cout << "Physical quadrature: " << (standard_quadrature ? "standard Gauss" : "auxiliary GLL") << std::endl; }
   a.AddDomainIntegrator(integrator); a.Assemble();
   OperatorPtr A; Vector X, B;
   a.FormLinearSystem(essential, x, b, A, X, B);
   StopWatch timer; timer.Start();
   vertex_patch::Preconditioner prec(fes, diffusion, ess_bdr, options);
   MFEM_STREAM_SYNC;
   timer.Stop(); real_t setup_time = timer.RealTime();
   prec.PrintStatistics(std::cout);
   Vector probe(B.Size());
   prec.Mult(B,probe); MFEM_STREAM_SYNC;
#ifdef MFEM_USE_CUDA
   if (profile && Device::Allows(Backend::CUDA_MASK)) { MFEM_GPU_CHECK(cudaProfilerStart()); }
#endif
   timer.Clear(); timer.Start();
   for (int i = 0; i < 3; ++i) { prec.Mult(B, probe); }
   MFEM_STREAM_SYNC;
   timer.Stop(); real_t application_time = timer.RealTime()/3.;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetOperator(*A); cg.SetPreconditioner(prec); cg.SetMaxIter(max_iterations);
   // CG's stopping norm is preconditioned; verify the true residual separately.
   cg.SetRelTol(tolerance * 0.01); cg.SetAbsTol(0.); cg.SetPrintLevel(1);
   timer.Clear(); timer.Start(); cg.Mult(B, X); MFEM_STREAM_SYNC; timer.Stop();
   real_t solve_time = timer.RealTime();
#ifdef MFEM_USE_CUDA
   if (profile && Device::Allows(Backend::CUDA_MASK)) { MFEM_GPU_CHECK(cudaProfilerStop()); }
#endif
   Vector residual(B.Size()); A->Mult(X, residual); residual -= B;
   const real_t bnorm = std::sqrt(InnerProduct(MPI_COMM_WORLD, B, B));
   const real_t relative = std::sqrt(InnerProduct(MPI_COMM_WORLD, residual,
                                                  residual)) / (bnorm > 0. ? bnorm : 1.);
   MPI_Allreduce(MPI_IN_PLACE, &setup_time, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &application_time, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &solve_time, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, MPI_COMM_WORLD);
   const auto size = fes.GlobalTrueVSize();
   if (!rank)
   {
      std::cout << "Unknowns: " << size << ", iterations: " << cg.GetNumIterations()
                << '\n'
                << "True relative residual: " << relative << '\n'
                << "Setup time (MPI max): " << setup_time << " s\n"
                << "Preconditioner application (MPI max, mean of 3): "
                << application_time << " s\n"
                << "Solve time (MPI max): " << solve_time << " s" << std::endl;
   }
   if (manufactured)
   {
      a.RecoverFEMSolution(X, b, x); const real_t error = x.ComputeL2Error(exact);
      if (!rank) { std::cout << "L2 error: " << error << std::endl; }
   }
   return cg.GetConverged() && std::isfinite(relative) &&
          relative <= tolerance ? 0 : 2;
}
#else
int main()
{
   std::cerr <<
             "vertex_patch requires MFEM_USE_MPI, MFEM_USE_LAPACK, and MFEM_USE_METIS_5.\n";
   return 1;
}
#endif
