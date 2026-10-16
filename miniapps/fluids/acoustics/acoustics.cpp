// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#include "acoustics.hpp"
#include <cmath>
#include <iomanip>

using namespace mfem;
using namespace acoustics;

int main(int argc, char *argv[])
{
   Mpi::Init(argc,argv);
   Hypre::Init();
   const MPI_Comm comm = MPI_COMM_WORLD;
   int n = 16, order = 2, method = 23, output_steps = 10;
   real_t dt = .01, final_time = .1;
   bool triangles = false, paraview = false, pc_test = false;
   std::string problem_name = "wave", mesh_file;
   Medium medium;
   SolverOptions options;

   OptionsParser args(argc,argv);
   args.AddOption(&n,"-n","--elements",
                  "Elements per direction in the periodic square.");
   args.AddOption(&mesh_file,"-m","--mesh",
                  "Conforming planar periodic mesh in [-2,2]^2.");
   args.AddOption(&order,"-o","--order","DG degree (at least one).");
   args.AddOption(&method,"-s","--ode-solver",
                  "4: RK4, 21: backward Euler, 22: SDIRK2, 23: SDIRK3.");
   args.AddOption(&dt,"-dt","--time-step","Fixed time step.");
   args.AddOption(&final_time,"-tf","--final-time","Final time.");
   args.AddOption(&medium.density,"-rho","--density",
                  "Positive background density.");
   args.AddOption(&medium.speed,"-c","--sound-speed","Positive sound speed.");
   args.AddOption(&problem_name,"-p","--problem","wave, multimode, or constant.");
   args.AddOption(&triangles,"-tri","--triangles","-quad","--quadrilaterals",
                  "Generated element type.");
   args.AddOption(&options.preconditioner,"-pc","--preconditioner",
                  "block-ilu, block-jacobi, block-gs, hypre-ilu, or none.");
   args.AddOption(&pc_test,"-pct","--pc-test","-no-pct","--no-pc-test",
                  "Solve (M/dt + K)x = b once with a fixed random RHS.");
   args.AddOption(&options.verbose,"-v","--verbose","-no-v","--no-verbose",
                  "Print step and stage diagnostics.");
   args.AddOption(&options.history,"-hist","--history",
                  "Per-stage solver history CSV.");
   args.AddOption(&options.export_prefix,"-ex","--export-prefix",
                  "Export stage matrices, base states, and RHS vectors.");
   args.AddOption(&paraview,"-pv","--paraview","-no-pv","--no-paraview",
                  "Write pressure and velocity to ParaView.");
   args.AddOption(&output_steps,"-os","--output-steps",
                  "ParaView output interval.");
   args.ParseCheck();

   MFEM_VERIFY(n >= 3 && order >= 1 && output_steps > 0 &&
               std::isfinite(dt) && dt > 0 && std::isfinite(final_time) && final_time >= 0 &&
               std::isfinite(medium.density) && medium.density > 0 &&
               std::isfinite(medium.speed) && medium.speed > 0,
               "Invalid physical or discretization options.");
   MFEM_VERIFY(problem_name == "wave" || problem_name == "multimode" ||
               problem_name == "constant","Unknown problem.");
   MFEM_VERIFY(method == 4 || method == 21 || method == 22 ||
               method == 23,"Unsupported ODE method.");

   MFEM_VERIFY(options.preconditioner == "block-ilu" ||
               options.preconditioner == "block-jacobi" ||
               options.preconditioner == "block-gs" ||
               options.preconditioner == "hypre-ilu" ||
               options.preconditioner == "none", "Unknown preconditioner.");

   Mesh mesh = mesh_file.empty() ? PeriodicMesh(n,triangles) : Mesh(mesh_file);
   MFEM_VERIFY(mesh.Dimension() == 2 && mesh.SpaceDimension() == 2 &&
               !mesh.Nonconforming(),
               "Only conforming planar 2D meshes are supported.");
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      const auto *tr = mesh.GetFaceElementTransformations(f);
      MFEM_VERIFY(tr && tr->Elem2No >= 0,"All boundaries must be periodic.");
   }

   ParMesh pmesh(comm,mesh);
   mesh.Clear();
   DG_FECollection fec(order,2);
   ParFiniteElementSpace fes(&pmesh,&fec,3,Ordering::byNODES);

   const double assembly_start = MPI_Wtime();
   SpatialOperator spatial(fes,medium);
   const double spatial_assembly = MPI_Wtime()-assembly_start;

   AcousticOperator flow(spatial,options);
   if (pc_test)
   {
      Vector rhs, state(fes.GetTrueVSize());
      RandomRightHandSide(fes,rhs);
      const double start = MPI_Wtime();
      flow.BeginStep(1,0,dt);
      flow.SolveSystem(dt,rhs,state);
      const SolverStats &stats = flow.Stats();
      double times[4] =
      {
         spatial_assembly+stats.assembly_time,
         stats.setup_time,
         stats.solve_time,
         MPI_Wtime()-start
      };
      double maximum[4];
      MPI_Allreduce(times,maximum,4,MPI_DOUBLE,MPI_MAX,comm);
      const auto elements = pmesh.GetGlobalNE();
      const auto dofs = fes.GlobalTrueVSize();
      const real_t rhs_norm = GlobalNorm(rhs,comm);

      if (Mpi::Root())
      {
         std::cout << std::setprecision(14)
                   << "RESULT pc_test=1 ranks=" << Mpi::WorldSize()
                   << " elements=" << elements << " dofs=" << dofs
                   << " order=" << order << " mass_shift=" << 1/dt
                   << " stages=" << stats.stages << " setups=" << stats.setups
                   << " krylov=" << stats.linear_iterations
                   << " rhs_norm=" << rhs_norm << " residual=" << stats.residual
                   << " relative_residual=" << stats.residual/rhs_norm
                   << " assembly_s=" << maximum[0] << " setup_s=" << maximum[1]
                   << " solve_s=" << maximum[2] << " elapsed_s=" << maximum[3] << '\n';
      }
      return 0;
   }

   auto ode = method == 4 ? ODESolver::SelectExplicit(4) :
              ODESolver::SelectImplicit(method);
   if (method == 4)
   {
      flow.SetExplicitMode();
   }
   ode->Init(flow);

   Problem problem(medium,problem_name);
   ParFiniteElementSpace pressure_fes(&pmesh,&fec);
   ParFiniteElementSpace velocity_fes(&pmesh,&fec,2,Ordering::byNODES);
   ParGridFunction solution(&fes), pressure(&pressure_fes),
                   velocity(&velocity_fes);
   VectorFunctionCoefficient initial(3,[&](const Vector &x, Vector &u)
   {
      problem.State(x,0,u);
   });
   solution.ProjectCoefficient(initial);
   Vector state;
   solution.GetTrueDofs(state);

   auto update_fields = [&]()
   {
      solution.SetFromTrueDofs(state);
      const int nd = fes.GetNDofs();
      for (int i = 0; i < nd; i++)
      {
         pressure(i) = solution(i);
         velocity(i) = solution(nd+i);
         velocity(nd+i) = solution(2*nd+i);
      }
   };
   update_fields();

   Vector initial_totals, totals;
   spatial.Conserved(state,initial_totals);
   const real_t initial_energy = spatial.Energy(state);

   std::unique_ptr<ParaViewDataCollection> output;
   if (paraview)
   {
      output.reset(new ParaViewDataCollection("Acoustics",&pmesh));
      output->SetPrefixPath("ParaView");
      output->SetLevelsOfDetail(order);
      output->RegisterField("pressure",&pressure);
      output->RegisterField("velocity",&velocity);
      output->SetCycle(0);
      output->SetTime(0);
      output->Save();
   }

   real_t time = 0;
   int steps = 0;
   const double start = MPI_Wtime();
   while (time < final_time)
   {
      const real_t target = std::min(final_time,(steps+1)*dt);
      real_t step = target-time;
      MFEM_VERIFY(step > 0,"Time step is too small to advance time.");
      flow.BeginStep(steps+1,time,step);
      ode->Step(state,time,step);
      time = target;
      steps++;
      CheckFinite(state);

      if (output && (steps%output_steps == 0 || time >= final_time))
      {
         update_fields();
         output->SetCycle(steps);
         output->SetTime(time);
         output->Save();
      }
   }

   const double elapsed = MPI_Wtime()-start;
   update_fields();

   FunctionCoefficient exact_pressure([&](const Vector &x)
   {
      Vector u;
      problem.State(x,time,u);
      return u(0);
   });
   VectorFunctionCoefficient exact_velocity(2,[&](const Vector &x, Vector &v)
   {
      Vector u;
      problem.State(x,time,u);
      v.SetSize(2);
      v(0) = u(1);
      v(1) = u(2);
   });
   const real_t pressure_error = pressure.ComputeL2Error(exact_pressure);
   const real_t velocity_error = velocity.ComputeL2Error(exact_velocity);
   const real_t error = std::hypot(pressure_error,velocity_error);
   const real_t energy = spatial.Energy(state);
   MFEM_VERIFY(std::isfinite(error) &&
               std::isfinite(energy),"Nonfinite acoustic diagnostics.");
   spatial.Conserved(state,totals);
   totals -= initial_totals;

   const SolverStats &stats = flow.Stats();
   double times[4] =
   {
      spatial_assembly+stats.assembly_time,
      stats.setup_time,
      stats.solve_time,
      elapsed
   };
   double maximum[4];
   MPI_Allreduce(times,maximum,4,MPI_DOUBLE,MPI_MAX,comm);
   const auto elements = pmesh.GetGlobalNE();
   const auto dofs = fes.GlobalTrueVSize();

   if (Mpi::Root())
   {
      std::cout << std::setprecision(14)
                << "RESULT ranks=" << Mpi::WorldSize()
                << " elements=" << elements << " dofs=" << dofs
                << " order=" << order << " method=" << method
                << " dt=" << dt << " time=" << time << " steps=" << steps
                << " pressure_error=" << pressure_error
                << " velocity_error=" << velocity_error
                << " error=" << error
                << " conservation=" << totals.Normlinf()
                << " initial_energy=" << initial_energy << " energy=" << energy
                << " stages=" << stats.stages << " setups=" << stats.setups
                << " krylov=" << stats.linear_iterations
                << " residual=" << stats.residual
                << " assembly_s=" << maximum[0] << " setup_s=" << maximum[1]
                << " solve_s=" << maximum[2] << " elapsed_s=" << maximum[3] << '\n';
   }

   return 0;
}
