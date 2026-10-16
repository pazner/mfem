// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#include "acoustics.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace mfem;
using namespace acoustics;

int main(int argc, char *argv[])
{
   Mpi::Init(argc,argv); Hypre::Init();
   const MPI_Comm comm = MPI_COMM_WORLD;
   int n = 16, order = 2, method = 23, offset = 3, output_steps = 10;
   real_t dt = .01, final_time = .1;
   bool triangles = false, paraview = false;
   std::string problem_name = "wave", mesh_file, save, reference;
   Medium medium; SolverOptions options;
   OptionsParser args(argc,argv);
   args.AddOption(&n,"-n","--elements",
                  "Elements per direction in the periodic square.");
   args.AddOption(&mesh_file,"-m","--mesh",
                  "Conforming planar periodic mesh in [-2,2]^2.");
   args.AddOption(&order,"-o","--order","DG degree (at least one).");
   args.AddOption(&method,"-s","--ode-solver",
                  "4: RK4, 21: backward Euler, 22: SDIRK2, 23: SDIRK3.");
   args.AddOption(&offset,"-q","--quadrature-offset",
                  "Quadrature order is 2*p + offset.");
   args.AddOption(&dt,"-dt","--time-step","Fixed time step.");
   args.AddOption(&final_time,"-tf","--final-time","Final time.");
   args.AddOption(&medium.density,"-rho","--density",
                  "Positive background density.");
   args.AddOption(&medium.speed,"-c","--sound-speed","Positive sound speed.");
   args.AddOption(&problem_name,"-p","--problem","wave or constant.");
   args.AddOption(&triangles,"-tri","--triangles","-quad","--quadrilaterals",
                  "Generated element type.");
   args.AddOption(&options.ilu,"-ilu","--ilu","-no-ilu","--no-ilu",
                  "Processor-block hypre ILU(0).");
   args.AddOption(&options.verbose,"-v","--verbose","-no-v","--no-verbose",
                  "Print step and stage diagnostics.");
   args.AddOption(&options.linear_rtol,"-lr","--linear-rtol",
                  "GMRES relative tolerance.");
   args.AddOption(&options.linear_max,"-lm","--linear-max",
                  "Maximum GMRES iterations.");
   args.AddOption(&options.krylov_dim,"-kd","--krylov-dim",
                  "GMRES restart dimension.");
   args.AddOption(&options.history,"-hist","--history",
                  "Per-stage solver history CSV.");
   args.AddOption(&options.export_prefix,"-ex","--export-prefix",
                  "Export stage matrices, base states, and RHS vectors.");
   args.AddOption(&save,"-save","--save-state",
                  "Save final per-rank true vectors.");
   args.AddOption(&reference,"-ref","--reference-state",
                  "Compare with saved vectors on identical mesh/partition.");
   args.AddOption(&paraview,"-pv","--paraview","-no-pv","--no-paraview",
                  "Write pressure and velocity to ParaView.");
   args.AddOption(&output_steps,"-os","--output-steps",
                  "ParaView output interval.");
   args.ParseCheck();
   MFEM_VERIFY(n >= 3 && order >= 1 && offset >= 0 && output_steps > 0 &&
               std::isfinite(dt) && dt > 0 && std::isfinite(final_time) && final_time >= 0 &&
               std::isfinite(medium.density) && medium.density > 0 &&
               std::isfinite(medium.speed) && medium.speed > 0,
               "Invalid physical or discretization options.");
   MFEM_VERIFY(std::isfinite(options.linear_rtol) && options.linear_rtol > 0 &&
               options.linear_rtol < 1 && options.linear_max > 0 && options.krylov_dim > 0,
               "Invalid solver options.");
   MFEM_VERIFY(problem_name == "wave" ||
               problem_name == "constant","Unknown problem.");
   MFEM_VERIFY(method == 4 || method == 21 || method == 22 ||
               method == 23,"Unsupported ODE method.");
   Mesh mesh = mesh_file.empty() ? PeriodicMesh(n,triangles) : Mesh(mesh_file);
   MFEM_VERIFY(mesh.Dimension() == 2 && mesh.SpaceDimension() == 2 &&
               !mesh.Nonconforming(),
               "Only conforming planar 2D meshes are supported.");
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      const auto *tr = mesh.GetFaceElementTransformations(f);
      MFEM_VERIFY(tr && tr->Elem2No >= 0,"All boundaries must be periodic.");
   }
   ParMesh pmesh(comm,mesh); mesh.Clear();
   DG_FECollection fec(order,2);
   ParFiniteElementSpace fes(&pmesh,&fec,3,Ordering::byNODES);
   ParFiniteElementSpace pressure_fes(&pmesh,&fec), velocity_fes(&pmesh,&fec,2,
                                                                 Ordering::byNODES);
   const double assembly_start = MPI_Wtime();
   SpatialOperator spatial(fes,medium,offset);
   const double spatial_assembly = MPI_Wtime()-assembly_start;
   AcousticOperator flow(spatial,options);
   auto ode = method == 4 ? ODESolver::SelectExplicit(4) :
              ODESolver::SelectImplicit(method);
   if (method == 4) { flow.SetExplicitMode(); }
   ode->Init(flow);
   Problem problem(medium,problem_name);
   ParGridFunction solution(&fes), pressure(&pressure_fes),
                   velocity(&velocity_fes);
   VectorFunctionCoefficient initial(3,[&](const Vector &x, Vector &u) { problem.State(x,0,u); });
   solution.ProjectCoefficient(initial);
   Vector state; solution.GetTrueDofs(state);
   CheckFinite(state,comm);
   auto update_fields = [&]()
   {
      solution.SetFromTrueDofs(state);
      const int nd = fes.GetNDofs();
      for (int i = 0; i < nd; i++)
      {
         pressure(i) = solution(i);
         velocity(i) = solution(nd+i); velocity(nd+i) = solution(2*nd+i);
      }
   };
   update_fields();
   Vector initial_totals, totals; spatial.Conserved(state,initial_totals);
   const real_t initial_energy = spatial.Energy(state);
   std::unique_ptr<ParaViewDataCollection> output;
   if (paraview)
   {
      output.reset(new ParaViewDataCollection("Acoustics",&pmesh));
      output->SetPrefixPath("ParaView"); output->SetLevelsOfDetail(order);
      output->RegisterField("pressure",&pressure);
      output->RegisterField("velocity",&velocity);
      output->SetCycle(0); output->SetTime(0); output->Save();
   }
   real_t time = 0; int steps = 0;
   const double start = MPI_Wtime();
   while (time < final_time)
   {
      const real_t target = std::min(final_time,(steps+1)*dt);
      real_t step = target-time;
      MFEM_VERIFY(step > 0,"Time step is too small to advance time.");
      flow.BeginStep(steps+1,time,step);
      ode->Step(state,time,step); time = target; steps++;
      CheckFinite(state,comm);
      if (output && (steps%output_steps == 0 || time >= final_time))
      {
         update_fields(); output->SetCycle(steps); output->SetTime(time); output->Save();
      }
   }
   const double elapsed = MPI_Wtime()-start;
   update_fields();
   FunctionCoefficient exact_pressure([&](const Vector &x)
   { Vector u; problem.State(x,time,u); return u(0); });
   VectorFunctionCoefficient exact_velocity(2,[&](const Vector &x, Vector &v)
   { Vector u; problem.State(x,time,u); v.SetSize(2); v(0) = u(1); v(1) = u(2); });
   const real_t pressure_error = pressure.ComputeL2Error(exact_pressure);
   const real_t velocity_error = velocity.ComputeL2Error(exact_velocity);
   const real_t error = std::hypot(pressure_error,velocity_error);
   const real_t energy = spatial.Energy(state);
   MFEM_VERIFY(std::isfinite(error) &&
               std::isfinite(energy),"Nonfinite acoustic diagnostics.");
   spatial.Conserved(state,totals); totals -= initial_totals;
   const int rank = Mpi::WorldRank();
   if (!save.empty())
   {
      std::ofstream file(save+"."+std::to_string(rank));
      file << state.Size() << '\n' << std::setprecision(17); state.Print(file,1);
      file.close(); VerifyCollectively(!file.fail(),comm,"Cannot write state file.");
   }
   real_t reference_error = -1;
   if (!reference.empty())
   {
      std::ifstream file(reference+"."+std::to_string(rank));
      int size = -1; file >> size;
      VerifyCollectively(!file.fail() && size == state.Size(),comm,
                         "Reference requires identical mesh, ordering, and partition.");
      Vector difference(size), weighted(size);
      for (int i = 0; i < size; i++) { file >> difference(i); }
      VerifyCollectively(!file.fail(),comm,"Invalid reference state.");
      CheckFinite(difference,comm);
      difference -= state; spatial.Mass().Mult(difference,weighted);
      real_t local = difference*weighted, global;
      MPI_Allreduce(&local,&global,1,MPITypeMap<real_t>::mpi_type,MPI_SUM,comm);
      reference_error = std::sqrt(std::max(real_t(0),global));
   }
   const SolverStats &stats = flow.Stats();
   double times[4] = {spatial_assembly+stats.assembly_time,stats.setup_time,stats.solve_time,elapsed},
                     maximum[4];
   MPI_Allreduce(times,maximum,4,MPI_DOUBLE,MPI_MAX,comm);
   const auto elements = pmesh.GetGlobalNE();
   const auto dofs = fes.GlobalTrueVSize();
   if (rank == 0)
   {
      std::cout << std::setprecision(14) << "RESULT ranks=" << Mpi::WorldSize()
                << " elements=" << elements << " dofs=" << dofs << " order=" << order
                << " method=" << method << " dt=" << dt << " time=" << time << " steps=" <<
                steps
                << " pressure_error=" << pressure_error << " velocity_error=" << velocity_error
                << " error=" << error << " reference_error=" << reference_error
                << " conservation=" << totals.Normlinf() << " initial_energy=" << initial_energy
                << " energy=" << energy << " stages=" << stats.stages << " setups=" <<
                stats.setups
                << " krylov=" << stats.linear_iterations << " residual=" << stats.residual
                << " assembly_s=" << maximum[0] << " setup_s=" << maximum[1]
                << " solve_s=" << maximum[2] << " elapsed_s=" << maximum[3] << '\n';
   }
   return 0;
}
