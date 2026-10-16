// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#include "acoustics.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace acoustics
{
SpatialOperator::SpatialOperator(ParFiniteElementSpace &f, const Medium &m)
   : Operator(f.GetTrueVSize()), fes(f), medium(m), flux(m), numerical_flux(flux),
     convection(numerical_flux,3,-1.0), residual(&f), mass(&f)
{
   MFEM_VERIFY(f.GetVDim() == 3 && f.GetOrdering() == Ordering::byNODES,
               "Expected three components in byNODES ordering.");
   MFEM_VERIFY(std::isfinite(m.density) && m.density > 0 &&
               std::isfinite(m.speed) && m.speed > 0, "Invalid acoustic medium.");
   AddIntegrators(residual);
   auto *mi = new VectorMassIntegrator;
   mi->SetVDim(3);
   mass.AddDomainIntegrator(mi);
   mass.Assemble();
   mass.Finalize();
   mass_matrix.reset(mass.ParallelAssemble());
   Vector zero(height);
   zero = 0.0;
   matrix = &dynamic_cast<HypreParMatrix &>(residual.GetGradient(zero));
}
void SpatialOperator::AddIntegrators(ParNonlinearForm &form)
{
   form.UseExternalIntegrators();
   form.AddDomainIntegrator(&convection);
   form.AddInteriorFaceIntegrator(&convection);
}
void SpatialOperator::Conserved(const Vector &u, Vector &totals) const
{
   Vector mu(height), local(3);
   Mass().Mult(u,mu);
   local = 0.0;
   for (int a = 0; a < 3; a++)
      for (int i = 0; i < fes.GetNDofs(); i++)
      {
         local(a) += mu(a*fes.GetNDofs()+i);
      }
   totals.SetSize(3);
   MPI_Allreduce(local.GetData(),totals.GetData(),3,MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,fes.GetComm());
}
void SpatialOperator::EnergyWeights(Vector &u) const
{
   const int n = fes.GetNDofs();
   for (int i = 0; i < n; i++)
   {
      u(i) /= medium.density*medium.speed*medium.speed;
      u(n+i) *= medium.density;
      u(2*n+i) *= medium.density;
   }
}
real_t SpatialOperator::Energy(const Vector &u) const
{
   Vector weighted(height);
   Mass().Mult(u,weighted);
   EnergyWeights(weighted);
   real_t local = .5*(u*weighted), global;
   MPI_Allreduce(&local,&global,1,MPITypeMap<real_t>::mpi_type,MPI_SUM,
                 fes.GetComm());
   return global;
}
AcousticOperator::AcousticOperator(SpatialOperator &s, const SolverOptions &o)
   : TimeDependentOperator(s.Height()), spatial(s), options(o), shift(0),
     mass_integrator(shift), stage_form(&s.Space()),
     mass_preconditioner(s.Mass(),HypreSmoother::Jacobi),
     mass_solver(s.Space().GetComm())
{
   SetImplicitVariableType(ImplicitVariableType::STATE);
   mass_integrator.SetVDim(3);
   spatial.AddIntegrators(stage_form);
   stage_form.AddDomainIntegrator(&mass_integrator);
   mass_solver.SetPreconditioner(mass_preconditioner);
   mass_solver.SetOperator(s.Mass());
   mass_solver.SetRelTol(1e-13);
   mass_solver.SetAbsTol(0);
   mass_solver.SetMaxIter(100);
   mass_solver.SetPrintLevel(-1);
   if (!options.history.empty())
   {
      bool ok = true;
      int rank;
      MPI_Comm_rank(s.Space().GetComm(),&rank);
      if (rank == 0)
      {
         std::ofstream file(options.history);
         file << "step,stage,system,time,mass_shift,reused,krylov,linear_residual,assembly_s,setup_s,solve_s\n";
         file.close();
         ok = !file.fail();
      }
      VerifyCollectively(ok,s.Space().GetComm(),"Cannot write solver history.");
   }
}
void AcousticOperator::SetPreconditionerFactory(PreconditionerFactory f)
{
   factory = std::move(f);
   cached_a = 0;
}
void AcousticOperator::SetLinearSolver(Solver &s)
{
   external_solver = &s;
   cached_a = 0;
}
void AcousticOperator::BeginStep(int number, real_t time, real_t dt)
{
   step_number = number;
   step_stage = 0;
   int rank;
   MPI_Comm_rank(spatial.Space().GetComm(),&rank);
   if (options.verbose && rank == 0)
   {
      mfem::out << "Step " << std::setw(6) << number << std::scientific
                << std::setprecision(6) << "  t = " << time << "  dt = " << dt << '\n';
   }
}
void AcousticOperator::Mult(const Vector &u, Vector &du) const
{
   const MPI_Comm comm = spatial.Space().GetComm();
   CheckFinite(u,comm);
   Vector rhs(height);
   spatial.Mult(u,rhs);
   rhs *= -1;
   mass_solver.iterative_mode = false;
   mass_solver.Mult(rhs,du);
   VerifyCollectively(mass_solver.GetConverged(),comm,"Mass solve failed.");
   CheckFinite(du,comm);
}
void AcousticOperator::ImplicitSolve(real_t a, const Vector &base,
                                     Vector &state)
{
   MFEM_VERIFY(ImplicitVarTypeIsState(),
               "Implicit solves require stage-state mode.");
   MFEM_VERIFY(std::isfinite(a) && a > 0,"Stage coefficient must be positive.");
   const MPI_Comm comm = spatial.Space().GetComm();
   int rank;
   MPI_Comm_rank(comm,&rank);
   CheckFinite(base,comm);
   // Indexed time targets can perturb an otherwise constant step by roundoff.
   const bool reused = cached_a > 0 && std::abs(a-cached_a) <=
                       64*std::numeric_limits<real_t>::epsilon()*std::max(a,cached_a);
   double assembly_seconds = 0, setup_seconds = 0;
   if (!reused)
   {
      gmres.reset();
      preconditioner.reset();
      shift.constant = 1/a;
      double start = MPI_Wtime();
      stage_matrix = &dynamic_cast<HypreParMatrix &>(stage_form.GetGradient(base));
      assembly_seconds = MPI_Wtime()-start;
      start = MPI_Wtime();
      if (external_solver)
      {
         external_solver->SetOperator(*stage_matrix);
      }
      else
      {
         gmres.reset(new GMRESSolver(comm));
         gmres->SetRelTol(options.linear_rtol);
         gmres->SetAbsTol(0);
         gmres->SetMaxIter(options.linear_max);
         gmres->SetKDim(options.krylov_dim);
         gmres->SetPrintLevel(-1);
         if (factory)
         {
            preconditioner = factory({spatial,*stage_matrix,shift.constant});
            VerifyCollectively(bool(preconditioner),comm,
                               "Preconditioner factory returned null.");
         }
         else if (options.ilu)
         {
#if MFEM_HYPRE_VERSION >= 21900
            auto ilu = std::unique_ptr<HypreILU>(new HypreILU);
            ilu->SetType(0);
            ilu->SetLevelOfFill(0);
            ilu->SetMaxIter(1);
            ilu->SetTol(0);
            ilu->SetPrintLevel(0);
            preconditioner = std::move(ilu);
#else
            MFEM_ABORT("HypreILU requires hypre >= 2.19; use --no-ilu.");
#endif
         }
         if (preconditioner)
         {
            gmres->SetPreconditioner(*preconditioner);
         }
         gmres->SetOperator(*stage_matrix);
         if (auto *hypre = dynamic_cast<HypreSolver *>(preconditioner.get()))
         {
            Vector zero(height), work(height);
            zero = 0.0;
            work = 0.0;
            hypre->Setup(zero,work);
         }
      }
      setup_seconds = MPI_Wtime()-start;
      stats.assembly_time += assembly_seconds;
      stats.setup_time += setup_seconds;
      stats.setups++;
      cached_a = a;
   }
   Vector rhs(height);
   spatial.Mass().Mult(base,rhs);
   rhs *= shift.constant;
   Solver &solver = external_solver ? *external_solver : *gmres;
   solver.iterative_mode = false;
   const double start = MPI_Wtime();
   solver.Mult(rhs,state);
   const double solve_seconds = MPI_Wtime()-start;
   const auto *iterative = dynamic_cast<IterativeSolver *>(&solver);
   VerifyCollectively(!iterative ||
                      iterative->GetConverged(),comm,"Linear solve failed.");
   CheckFinite(state,comm);
   Vector residual(height);
   stage_matrix->Mult(state,residual);
   residual -= rhs;
   stats.residual = GlobalNorm(residual,comm);
   const int iterations = iterative ? iterative->GetNumIterations() : -1;
   if (iterative)
   {
      stats.linear_iterations += iterations;
   }
   stats.solve_time += solve_seconds;
   stats.stages++;
   step_stage++;
   if (options.verbose && rank == 0)
   {
      mfem::out << "  Stage " << std::setw(2) << step_stage
                << "  Krylov iterations = " << iterations << "  residual = "
                << stats.residual << "  setup reused = " << reused << '\n';
   }
   if (!options.history.empty())
   {
      double times[3] = {assembly_seconds,setup_seconds,solve_seconds}, maximum[3];
      MPI_Allreduce(times,maximum,3,MPI_DOUBLE,MPI_MAX,comm);
      bool ok = true;
      if (rank == 0)
      {
         std::ofstream file(options.history,std::ios::app);
         file << std::setprecision(17) << step_number << ',' << step_stage << ','
              << stats.stages << ',' << GetTime() << ',' << shift.constant << ','
              << reused << ',' << iterations << ',' << stats.residual << ','
              << maximum[0] << ',' << maximum[1] << ',' << maximum[2] << '\n';
         file.close();
         ok = !file.fail();
      }
      VerifyCollectively(ok,comm,"Cannot write solver history.");
   }
   if (!options.export_prefix.empty())
   {
      const std::string name = options.export_prefix+"-stage"+std::to_string(
                                  stats.stages);
      bool ok = true;
      for (const auto &entry :
           {
              std::make_pair("-base.",&base),std::make_pair("-rhs.",
                                                            static_cast<const Vector *>(&rhs))
           })
      {
         std::ofstream file(name+entry.first+std::to_string(rank));
         file << std::setprecision(17);
         entry.second->Print(file,1);
         file.close();
         ok = ok && !file.fail();
      }
      if (rank == 0)
      {
         std::ofstream file(name+"-meta.txt");
         file << std::setprecision(17) << "mass_shift " << shift.constant
              << "\ntime " << GetTime() << "\nstep " << step_number
              << "\nstage " << step_stage << "\nreused " << reused << '\n';
         file.close();
         ok = ok && !file.fail();
      }
      VerifyCollectively(ok,comm,"Cannot export stage vectors or metadata.");
      stage_matrix->Print((name+"-A").c_str());
   }
}
}
