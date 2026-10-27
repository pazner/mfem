// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#include "flow.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace mfem;
using namespace compressible;

int main(int argc, char *argv[])
{
   Mpi::Init(argc,argv); Hypre::Init();
   int n = 16, order = 2, method = 23, offset = 3, output_steps = 10;
   real_t dt = .01, final_time = .1, penalty = 10;
   bool triangles = false, paraview = false;
   std::string problem_name = "vortex", mesh_file, save, reference;
   Gas gas; SolverOptions solver_options;
   OptionsParser args(argc,argv);
   args.AddOption(&n,"-n","--elements",
                  "Elements per coordinate direction in the periodic square.");
   args.AddOption(&mesh_file,"-m","--mesh",
                  "Optional conforming 2D periodic mesh in [-2,2]^2.");
   args.AddOption(&order,"-o","--order","DG polynomial degree (at least one).");
   args.AddOption(&method,"-s","--ode-solver",
                  "21: backward Euler, 22: SDIRK2, 23: SDIRK3, 4: explicit RK4 reference.");
   args.AddOption(&offset,"-q","--quadrature-offset",
                  "Integration order is 2*p + offset.");
   args.AddOption(&dt,"-dt","--time-step","Fixed time step.");
   args.AddOption(&final_time,"-tf","--final-time","Final time.");
   args.AddOption(&problem_name,"-p","--problem",
                  "vortex, constant, or manufactured.");
   args.AddOption(&gas.gamma,"-g","--gamma","Ratio of specific heats.");
   args.AddOption(&gas.gas_constant,"-rg","--gas-constant",
                  "Specific gas constant.");
   args.AddOption(&gas.viscosity,"-mu","--viscosity",
                  "Dynamic viscosity (zero selects Euler).");
   args.AddOption(&gas.prandtl,"-pr","--prandtl","Prandtl number.");
   args.AddOption(&penalty,"-ip","--penalty","Interior penalty multiplier.");
   args.AddOption(&triangles,"-tri","--triangles","-quad","--quadrilaterals",
                  "Generated mesh element type.");
   args.AddOption(&solver_options.prec_bool,"-ilu","--ilu","-agglom","--agglom",
                  "Choose the preconditioner.");
   args.AddOption(&solver_options.verbose,"-v","--verbose","-no-v","--no-verbose",
                  "Print step headings and per-stage Newton/GMRES iterations.");
   args.AddOption(&solver_options.newton_rtol,"-nr","--newton-rtol",
                  "Newton relative tolerance.");
   args.AddOption(&solver_options.newton_atol,"-na","--newton-atol",
                  "Newton absolute tolerance.");
   args.AddOption(&solver_options.newton_max,"-nm","--newton-max",
                  "Maximum Newton iterations.");
   args.AddOption(&solver_options.linear_rtol,"-lr","--linear-rtol",
                  "GMRES relative tolerance.");
   args.AddOption(&solver_options.linear_max,"-lm","--linear-max",
                  "Maximum GMRES iterations.");
   args.AddOption(&solver_options.krylov_dim,"-kd","--krylov-dim",
                  "GMRES restart dimension.");
   args.AddOption(&solver_options.export_prefix,"-ex","--export-prefix",
                  "Export every Newton matrix, state and RHS to this prefix.");
   args.AddOption(&solver_options.history,"-hist","--history",
                  "CSV file for per-Newton-system solver history.");
   args.AddOption(&save,"-save","--save-state",
                  "Save final per-rank true vectors to this prefix.");
   args.AddOption(&reference,"-ref","--reference-state",
                  "Compare with saved vectors on the identical mesh and partition.");
   args.AddOption(&paraview,"-pv","--paraview","-no-pv","--no-paraview",
                  "Write ParaView output.");
   args.AddOption(&output_steps,"-os","--output-steps",
                  "ParaView output interval.");
   args.ParseCheck();
   MFEM_VERIFY(n >= 3 && order >= 1 && offset >= 0 && dt > 0 && final_time >= 0 &&
               penalty > 0 && gas.gamma > 1 && gas.gas_constant > 0 &&
               gas.viscosity >= 0 && gas.prandtl > 0 && output_steps > 0,
               "Invalid physical or discretization options.");
   MFEM_VERIFY(solver_options.newton_rtol > 0 && solver_options.newton_atol >= 0 &&
               solver_options.linear_rtol > 0 && solver_options.newton_max > 0 &&
               solver_options.linear_max > 0 && solver_options.krylov_dim > 0,
               "Invalid solver options.");
   MFEM_VERIFY(problem_name == "vortex" || problem_name == "constant" ||
               problem_name == "manufactured", "Unknown problem.");
   MFEM_VERIFY(method == 21 || method == 22 || method == 23 ||
               method == 4,"Unsupported ODE method.");
   Mesh mesh = mesh_file.empty() ? PeriodicMesh(n,triangles) : Mesh(mesh_file);
   MFEM_VERIFY(mesh.Dimension() == 2 && mesh.SpaceDimension() == 2 &&
               !mesh.Nonconforming(),
               "Only conforming planar 2D meshes are supported.");
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      auto *tr = mesh.GetFaceElementTransformations(f);
      MFEM_VERIFY(tr && tr->Elem2No >= 0,"All boundaries must be periodic.");
   }
   ParMesh pmesh(MPI_COMM_WORLD,mesh); mesh.Clear();
   DG_FECollection fec(order,2);
   ParFiniteElementSpace fes(&pmesh,&fec,4,Ordering::byNODES);
   Problem problem(gas,problem_name);
   SpatialOperator spatial(fes,gas,problem,offset,penalty);
   FlowOperator flow(spatial,solver_options);
   auto ode = method == 4 ? ODESolver::SelectExplicit(4) :
              ODESolver::SelectImplicit(method);
   if (method == 4) { flow.SetExplicitMode(); }
   ode->Init(flow);
   ParGridFunction solution(&fes);
   VectorFunctionCoefficient initial(4,[&](const Vector &x, Vector &u) { problem.State(x,0,u); });
   solution.ProjectCoefficient(initial);
   Vector state; solution.GetTrueDofs(state);
   MFEM_VERIFY(spatial.Admissible(state),
               "Initial projection is not physical; refine the mesh.");
   Vector initial_totals, totals;
   spatial.Conserved(state,initial_totals);
   std::unique_ptr<ParaViewDataCollection> output;
   if (paraview)
   {
      output.reset(new ParaViewDataCollection("Compressible",&pmesh));
      output->SetPrefixPath("ParaView"); output->SetLevelsOfDetail(order);
      output->RegisterField("conservative_state",&solution);
      output->SetCycle(0); output->SetTime(0); output->Save();
   }
   real_t time = 0;
   int steps = 0;
   const double start = MPI_Wtime();
   while (time < final_time)
   {
      // Compute targets from the step index, not accumulated additions. This
      // prevents a roundoff-sized extra implicit solve at the final time.
      const real_t target = std::min(final_time,(steps+1)*dt);
      real_t step = target-time;
      MFEM_VERIFY(step > 0.0, "Time step is too small to advance time.");
      flow.BeginStep(steps+1,time,step);
      ode->Step(state,time,step);
      time = target;
      MFEM_VERIFY(spatial.Admissible(state),
                  "Time integrator produced a nonphysical state.");
      steps++;
      solution.SetFromTrueDofs(state);
      if (output && (steps%output_steps == 0 || time >= final_time))
      { output->SetCycle(steps); output->SetTime(time); output->Save(); }
   }
   double elapsed = MPI_Wtime()-start, max_elapsed;
   MPI_Allreduce(&elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
   VectorFunctionCoefficient exact(4,[&](const Vector &x, Vector &u) { problem.State(x,time,u); });
   // A viscous unforced vortex has no analytic reference here.
   real_t error = -1;
   if (problem_name != "vortex" || gas.viscosity == 0) { error = solution.ComputeL2Error(exact); }
   spatial.Conserved(state,totals); totals -= initial_totals;
   const int rank = Mpi::WorldRank();
   if (!save.empty())
   {
      std::ofstream state_file(save+"."+std::to_string(rank));
      MFEM_VERIFY(state_file.good(),"Cannot write state file.");
      state_file << state.Size() << '\n' << std::setprecision(17);
      state.Print(state_file,1);
   }
   real_t reference_error = -1;
   if (!reference.empty())
   {
      std::ifstream in(reference+"."+std::to_string(rank));
      int size = -1; in >> size;
      MFEM_VERIFY(size == state.Size(),
                  "Reference requires identical mesh, ordering, and partition.");
      Vector difference(state.Size()), weighted(state.Size());
      for (int i = 0; i < size; i++) { in >> difference(i); }
      MFEM_VERIFY(in.good(),"Invalid reference state.");
      difference -= state; spatial.Mass().Mult(difference,weighted);
      real_t local = difference*weighted, global;
      MPI_Allreduce(&local,&global,1,MPITypeMap<real_t>::mpi_type,MPI_SUM,
                    MPI_COMM_WORLD);
      reference_error = std::sqrt(global);
   }
   const SolverStats &stats = flow.Stats();
   double times[3] = {stats.assembly_time,stats.setup_time,stats.solve_time},
                     max_times[3];
   MPI_Allreduce(times,max_times,3,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
   const auto global_elements = pmesh.GetGlobalNE();
   const auto global_dofs = fes.GlobalTrueVSize();
   if (rank == 0)
   {
      std::cout << std::setprecision(14)
                << "RESULT ranks=" << Mpi::WorldSize() << " elements=" << global_elements
                << " dofs=" << global_dofs << " order=" << order << " method=" << method
                << " dt=" << dt << " time=" << time << " steps=" << steps
                << " error=" << error << " reference_error=" << reference_error
                << " conservation=" << totals.Normlinf() << " newton=" <<
                stats.newton_iterations
                << " krylov=" << stats.linear_iterations << " residual=" << stats.residual
                << " assembly_s=" << max_times[0] << " setup_s=" << max_times[1]
                << " solve_s=" << max_times[2] << " elapsed_s=" << max_elapsed << '\n';
   }
   return 0;
}
