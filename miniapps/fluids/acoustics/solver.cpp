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
ElementBlockPreconditioner::ElementBlockPreconditioner(
   ParFiniteElementSpace &fes, const std::string &type)
   : Solver(fes.GetTrueVSize())
{
   MFEM_VERIFY(fes.IsDGSpace() && fes.GetVSize() == height,
               "Element blocks require a discontinuous finite element space.");
   permutation.SetSize(height);
   inverse_permutation.SetSize(height);
   inverse_permutation = -1;
   int block_size = 0;
   int next = 0;
   bool uniform = true;
   Array<int> dofs;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      fes.GetElementVDofs(e,dofs);
      if (e == 0)
      {
         block_size = dofs.Size();
      }
      uniform = uniform && dofs.Size() == block_size;
      for (int dof : dofs)
      {
         MFEM_VERIFY(dof >= 0, "Expected unsigned DG degrees of freedom.");
         const int tdof = fes.GetLocalTDofNumber(dof);
         MFEM_VERIFY(tdof >= 0 && tdof < height &&
                     inverse_permutation[tdof] == -1,
                     "Element blocks must partition the local true degrees of freedom.");
         permutation[next] = tdof;
         inverse_permutation[tdof] = next++;
      }
   }
   MFEM_VERIFY(next == height, "Incomplete element block partition.");
   VerifyCollectively(uniform,fes.GetComm(),
                      "Element preconditioners require uniform element sizes on each rank.");
   // Empty ranks never apply BlockILU, which requires a nonempty local matrix.
   block_size = std::max(1,block_size);
   if (type == "block-ilu")
   {
      local_solver.reset(new BlockILU(block_size));
   }
   else if (type == "block-jacobi")
   {
      local_solver.reset(new BlockJacobi(block_size));
   }
   else
   {
      MFEM_VERIFY(type == "block-gs", "Unknown element preconditioner.");
      local_solver.reset(new BlockGaussSeidel(block_size));
   }
   rhs.SetSize(height);
   solution.SetSize(height);
}

void ElementBlockPreconditioner::SetOperator(const Operator &op)
{
   const auto *parallel = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(parallel && op.Height() == height && op.Width() == width,
               "Expected a parallel matrix on the element space.");
   if (height == 0)
   {
      return;
   }
   SparseMatrix diagonal;
   parallel->GetDiag(diagonal);
   MFEM_VERIFY(diagonal.Height() == height && diagonal.Width() == width,
               "Expected matching local row and column partitions.");
   const int *I = diagonal.HostReadI();
   const int *J = diagonal.HostReadJ();
   const real_t *values = diagonal.HostReadData();
   SparseMatrix permuted(height,width);
   for (int i = 0; i < height; i++)
   {
      const int row = permutation[i];
      for (int k = I[row]; k < I[row+1]; k++)
      {
         permuted.Add(i,inverse_permutation[J[k]],values[k]);
      }
   }
   permuted.Finalize();
   local_solver->SetOperator(permuted);
}

void ElementBlockPreconditioner::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(b.Size() == height, "Invalid element right-hand side size.");
   for (int i = 0; i < height; i++)
   {
      rhs(i) = b(permutation[i]);
   }
   if (height > 0)
   {
      local_solver->Mult(rhs,solution);
   }
   x.SetSize(height);
   for (int i = 0; i < height; i++)
   {
      x(permutation[i]) = solution(i);
   }
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
   CheckFinite(base,spatial.Space().GetComm());
   Vector rhs(height);
   spatial.Mass().Mult(base,rhs);
   rhs /= a;
   SolveShifted(a,rhs,state,&base);
}
void AcousticOperator::SolveSystem(real_t a, const Vector &rhs, Vector &state)
{
   SolveShifted(a,rhs,state,nullptr);
}
void AcousticOperator::SolveShifted(real_t a, const Vector &rhs, Vector &state,
                                    const Vector *base)
{
   MFEM_VERIFY(std::isfinite(a) && a > 0,"Stage coefficient must be positive.");
   MFEM_VERIFY(rhs.Size() == height,"Invalid right-hand side size.");
   const MPI_Comm comm = spatial.Space().GetComm();
   int rank;
   MPI_Comm_rank(comm,&rank);
   CheckFinite(rhs,comm);
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
      // This linear form has the same gradient at every state.
      stage_matrix = &dynamic_cast<HypreParMatrix &>(stage_form.GetGradient(rhs));
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
         else if (options.preconditioner == "hypre-ilu")
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
            MFEM_ABORT("hypre-ilu requires hypre >= 2.19; choose another preconditioner.");
#endif
         }
         else if (options.preconditioner != "none")
         {
            preconditioner.reset(new ElementBlockPreconditioner(
                                    spatial.Space(),options.preconditioner));
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
              std::make_pair("-base.",base),std::make_pair("-rhs.",&rhs)
           })
      {
         if (!entry.second)
         {
            continue;
         }
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
