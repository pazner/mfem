// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#include "flow.hpp"
#include "../../dg-agglomeration/mg_agglom.hpp"
#include "../../dg-agglomeration/partition.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace compressible
{
SpatialOperator::SpatialOperator(ParFiniteElementSpace &f, const Gas &g,
                                 const Problem &p, int offset_, real_t penalty)
   : Operator(f.GetTrueVSize()), fes(f), gas(g), offset(offset_), flux(2,g.gamma),
     numerical_flux(flux,g), convection(numerical_flux,offset,-1.0),
     diffusion(g,offset,penalty), residual(&f), mass(&f), problem(p)
{
   MFEM_VERIFY(f.GetVDim() == 4 && f.GetOrdering() == Ordering::byNODES,
               "Expected four conservative components in byNODES ordering.");
   AddIntegrators(residual);
   auto *mi = new VectorMassIntegrator;
   mi->SetVDim(4);
   mass.AddDomainIntegrator(mi);
   mass.Assemble(); mass.Finalize(); mass_matrix.reset(mass.ParallelAssemble());
}
void SpatialOperator::AddIntegrators(ParNonlinearForm &form)
{
   form.UseExternalIntegrators();
   form.AddDomainIntegrator(&convection);
   form.AddInteriorFaceIntegrator(&convection);
   if (gas.viscosity != 0.0)
   {
      form.AddDomainIntegrator(&diffusion);
      form.AddInteriorFaceIntegrator(&diffusion);
   }
}
void SpatialOperator::Mult(const Vector &u, Vector &r) const
{
   residual.Mult(u,r);
   if (problem.HasSource())
   {
      VectorFunctionCoefficient source(4,[this](const Vector &x, Vector &s)
      { problem.Source(x,time,s); });
      auto *integrator = new VectorDomainLFIntegrator(source);
      // Match the residual's integration rule, including triangle geometry.
      for (int e = 0; e < fes.GetNE(); e++)
      {
         const FiniteElement &fe = *fes.GetFE(e);
         integrator->SetIntRule(&IntRules.Get(fe.GetGeomType(),2*fe.GetOrder()+offset));
         Vector local;
         integrator->AssembleRHSElementVect(fe,*fes.GetElementTransformation(e),local);
         Array<int> dofs; fes.GetElementVDofs(e,dofs);
         r.AddElementVector(dofs,-1.0,local);
      }
      delete integrator;
   }
}
Operator &SpatialOperator::GetGradient(const Vector &u) const
{ return residual.GetGradient(u); }

bool SpatialOperator::Admissible(const Vector &x) const
{
   ParGridFunction u(&fes);
   u.SetFromTrueDofs(x);
   int ok = 1;
   auto check = [&](const Vector &v)
   {
      if (!std::isfinite(v(0)) || v(0) <= 0.0) { ok = 0; return; }
      const real_t p = gas.Pressure(v);
      if (!std::isfinite(p) || p <= 0.0) { ok = 0; }
   };
   Vector val(4);
   for (int e = 0; e < fes.GetNE(); e++)
   {
      const FiniteElement &fe = *fes.GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe.GetGeomType(),
                                               2*fe.GetOrder()+offset);
      for (int j = 0; j < ir.GetNPoints(); j++)
      { u.GetVectorValue(e,ir.IntPoint(j),val); check(val); }
   }
   // Check both local traces, including local traces of shared faces. A global
   // reduction then covers remote traces without evaluating invalid fluxes.
   ParMesh &mesh = *fes.GetParMesh();
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      auto *tr = mesh.GetFaceElementTransformations(f);
      if (!tr || tr->Elem1No < 0) { continue; }
      const int p = fes.GetFE(tr->Elem1No)->GetOrder();
      const IntegrationRule &ir = IntRules.Get(tr->GetGeometryType(),2*p+offset);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         tr->SetAllIntPoints(&ir.IntPoint(j));
         u.GetVectorValue(tr->Elem1No,tr->GetElement1IntPoint(),val); check(val);
         if (tr->Elem2No >= 0 && tr->Elem2No < fes.GetNE())
         { u.GetVectorValue(tr->Elem2No,tr->GetElement2IntPoint(),val); check(val); }
      }
   }
   int all_ok;
   MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,fes.GetComm());
   return all_ok != 0;
}
void SpatialOperator::Conserved(const Vector &u, Vector &totals) const
{
   Vector mu(height), local(4); Mass().Mult(u,mu); local = 0.0;
   for (int a = 0; a < 4; a++)
      for (int i = 0; i < fes.GetNDofs(); i++) { local(a) += mu(a*fes.GetNDofs()+i); }
   totals.SetSize(4);
   MPI_Allreduce(local.GetData(),totals.GetData(),4,MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,fes.GetComm());
}
StageOperator::StageOperator(SpatialOperator &s, SolverStats &stats_)
   : Operator(s.Height()), spatial(s), shift(0.0), mass_integrator(shift),
     form(&s.Space()), base_state(s.Height()), stats(stats_)
{
   mass_integrator.SetVDim(4);
   spatial.AddIntegrators(form);
   form.AddDomainIntegrator(&mass_integrator);
}
void StageOperator::Configure(real_t a, const Vector &base)
{
   MFEM_VERIFY(a > 0.0, "Stage coefficient must be positive.");
   shift.constant = 1/a;
   base_state = base;
}
void StageOperator::Mult(const Vector &u, Vector &r) const
{
   spatial.Mult(u,r);
   Vector difference(u), mu(height); difference -= base_state;
   spatial.Mass().Mult(difference,mu); r.Add(shift.constant,mu);
}
Operator &StageOperator::GetGradient(const Vector &u) const
{
   state = &u;
   const double start = MPI_Wtime();
   Operator &j = form.GetGradient(u);
   stats.assembly_time += MPI_Wtime()-start;
   return j;
}

namespace
{
class LinearSolver : public Solver
{
   SpatialOperator &spatial;
   StageOperator &stage;
   SolverOptions options;
   SolverStats &stats;
   PreconditionerFactory factory;
   std::unique_ptr<Solver> preconditioner;
   std::unique_ptr<GMRESSolver> gmres;
   Solver *external;
   int system = 0;
   double setup_seconds = 0.0;
   std::string system_name;
public:
   LinearSolver(SpatialOperator &s, StageOperator &st, const SolverOptions &o,
                SolverStats &stat, PreconditionerFactory f, Solver *ext)
      : Solver(s.Height()), spatial(s), stage(st), options(o), stats(stat),
        factory(std::move(f)), gmres(std::unique_ptr<GMRESSolver>(new GMRESSolver(s.Space().GetComm()))), external(ext)
   {
      gmres->SetRelTol(o.linear_rtol); gmres->SetAbsTol(0.0);
      gmres->SetMaxIter(o.linear_max); gmres->SetKDim(o.krylov_dim);
      gmres->SetPrintLevel(-1);
   }
   void SetOperator(const Operator &op) override
   {
      gmres.reset(); 
      gmres.reset(new GMRESSolver(spatial.Space().GetComm()));
      gmres->SetRelTol(options.linear_rtol);
      gmres->SetAbsTol(0);
      gmres->SetMaxIter(options.linear_max);
      gmres->SetKDim(options.krylov_dim);
      gmres->SetPrintLevel(-1);
      const double start = MPI_Wtime();
      if (external) { external->SetOperator(op); }
      else
      {
         auto &matrix = const_cast<HypreParMatrix &>
                        (dynamic_cast<const HypreParMatrix &>(op));
         if (factory)
         {
            preconditioner = factory({spatial,stage.State(),matrix,stage.Shift()});
            MFEM_VERIFY(preconditioner, "Preconditioner factory returned null.");
         }
         else if (options.prec_bool)
         {
#if MFEM_HYPRE_VERSION >= 21900
            auto ilu = std::unique_ptr<HypreILU>(new HypreILU);
            ilu->SetType(0); ilu->SetLevelOfFill(0); ilu->SetMaxIter(1);
            ilu->SetTol(0.0); ilu->SetPrintLevel(0);
            preconditioner = std::move(ilu);
#else
            MFEM_ABORT("HypreILU requires hypre >= 2.19; use --no-ilu.");
#endif
         }
         else
         {
            int dim = spatial.Space().GetMesh()->Dimension();
            int ne = spatial.Space().GetNE();
            int nl = std::ceil(std::log(ne)/std::log(4.0));
            int ncoarse = pow(2, dim); 
            preconditioner.reset(new SmoothedAggregationGMG(spatial.Space(), matrix, ncoarse, nl, false, SmoothedAggregationGMG::SmootherType::BlockGaussSeidel));
         }
         if (preconditioner && !options.prec_bool)
         {  
            gmres->SetOperator(op);
            gmres->SetPreconditioner(*preconditioner); 
         }
         else if (preconditioner && options.prec_bool)
         { 
            gmres->SetPreconditioner(*preconditioner); 
            gmres->SetOperator(op);
         }
         else
         {
            gmres->SetOperator(op);
         }
         if (auto *hypre = dynamic_cast<HypreSolver *>(preconditioner.get()))
         {
            Vector zero(height), work(height); zero = 0.0; work = 0.0;
            hypre->Setup(zero,work);
         }
      }
      setup_seconds = MPI_Wtime()-start;
      stats.setup_time += setup_seconds;
      system_name = options.export_prefix+"-stage"+std::to_string(stats.stages)+
                    "-newton"+std::to_string(system++);
      if (!options.export_prefix.empty())
      {
         const std::string &name = system_name;
         dynamic_cast<const HypreParMatrix &>(op).Print((name+"-A").c_str());
         int rank; MPI_Comm_rank(spatial.Space().GetComm(),&rank);
         std::ofstream out(name+"-state."+std::to_string(rank));
         MFEM_VERIFY(out.good(), "Cannot export state vector.");
         out << std::setprecision(17); stage.State().Print(out,1);
         if (rank == 0)
         {
            std::ofstream meta(name+"-meta.txt");
            MFEM_VERIFY(meta.good(), "Cannot export system metadata.");
            meta << std::setprecision(17) << "mass_shift " << stage.Shift() << '\n';
         }
      }
   }
   void Mult(const Vector &b, Vector &x) const override
   {
      const double start = MPI_Wtime();
      Solver &solver = external ? *external : const_cast<GMRESSolver &>(*(gmres.get()));
      solver.iterative_mode = false;
      solver.Mult(b,x);
      const auto *iterative = dynamic_cast<IterativeSolver *>(&solver);
      int ok = iterative ? iterative->GetConverged() : 1, all_ok;
      MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,spatial.Space().GetComm());
      MFEM_VERIFY(all_ok,
                  "Linear solve failed; reduce dt or improve the preconditioner.");
      if (iterative) { stats.linear_iterations += iterative->GetNumIterations(); }
      const double solve_seconds = MPI_Wtime()-start;
      stats.solve_time += solve_seconds;
      if (!options.history.empty())
      {
         double local_times[2] = {setup_seconds,solve_seconds}, max_times[2];
         MPI_Allreduce(local_times,max_times,2,MPI_DOUBLE,MPI_MAX,
                       spatial.Space().GetComm());
         real_t local_norm = b*b, global_norm;
         MPI_Allreduce(&local_norm,&global_norm,1,MPITypeMap<real_t>::mpi_type,MPI_SUM,
                       spatial.Space().GetComm());
         int rank; MPI_Comm_rank(spatial.Space().GetComm(),&rank);
         if (rank == 0)
         {
            std::ofstream history(options.history,std::ios::app);
            MFEM_VERIFY(history.good(), "Cannot write solver history.");
            history << std::setprecision(17) << stats.stages << ',' << system-1 << ','
                    << spatial.GetTime() << ',' << stage.Shift() << ',' << std::sqrt(
                       global_norm) << ','
                    << (iterative ? iterative->GetNumIterations() : -1) << ','
                    << (iterative ? iterative->GetFinalNorm() : -1) << ','
                    << max_times[0] << ',' << max_times[1] << '\n';
         }
      }
      if (!options.export_prefix.empty())
      {
         const std::string name = options.export_prefix+"-stage"+std::to_string(
                                     stats.stages)+
                                  "-newton"+std::to_string(system-1);
         int rank; MPI_Comm_rank(spatial.Space().GetComm(),&rank);
         std::ofstream rhs(name+"-rhs."+std::to_string(rank));
         MFEM_VERIFY(rhs.good(), "Cannot export right-hand side.");
         rhs << std::setprecision(17); b.Print(rhs,1);
      }
   }
};
class FlowNewton : public NewtonSolver
{
   SpatialOperator &spatial;
public:
   FlowNewton(SpatialOperator &s) : NewtonSolver(s.Space().GetComm()), spatial(s) { }
   real_t ComputeScalingFactor(const Vector &x, const Vector &b) const override
   {
      Vector trial(x.Size()), residual(x.Size());
      const real_t norm = Norm(r);
      real_t alpha = 1.0;
      for (int i = 0; i < 24; i++, alpha *= .5)
      {
         add(x,-alpha,c,trial);
         if (!spatial.Admissible(trial)) { continue; }
         oper->Mult(trial,residual);
         if (b.Size() == residual.Size()) { residual -= b; }
         const real_t trial_norm = Norm(residual);
         if (std::isfinite(trial_norm) && trial_norm <= (1-1e-4*alpha)*norm) { return alpha; }
      }
      return 0.0;
   }
};
}
FlowOperator::FlowOperator(SpatialOperator &s, const SolverOptions &o)
   : TimeDependentOperator(s.Height()), spatial(s), options(o), stage(s,stats)
{
   SetImplicitVariableType(ImplicitVariableType::STATE);
   if (!options.history.empty())
   {
      int rank; MPI_Comm_rank(spatial.Space().GetComm(),&rank);
      if (rank == 0)
      {
         std::ofstream history(options.history);
         MFEM_VERIFY(history.good(), "Cannot write solver history.");
         history <<
                 "stage,newton,time,mass_shift,nonlinear_residual,krylov,linear_residual,setup_s,solve_s\n";
      }
   }
}
void FlowOperator::BeginStep(int step_number, real_t time, real_t dt)
{
   step_stage = 0;
   if (!options.verbose) { return; }
   int rank; MPI_Comm_rank(spatial.Space().GetComm(),&rank);
   if (rank == 0)
   {
      std::ostringstream line;
      line << "Step " << std::setw(6) << step_number
           << std::scientific << std::setprecision(6)
           << "  t = " << std::setw(13) << time
           << "  dt = " << std::setw(13) << dt;
      mfem::out << line.str() << std::endl;
   }
}
void FlowOperator::Mult(const Vector &u, Vector &du) const
{
   spatial.SetTime(GetTime());
   MFEM_VERIFY(spatial.Admissible(u), "Nonphysical explicit state.");
   Vector r(height); spatial.Mult(u,r); r *= -1;
   // Consistent block-diagonal DG mass inverse through MFEM's existing solver.
   CGSolver cg(spatial.Space().GetComm());
   HypreSmoother diagonal(spatial.Mass(),HypreSmoother::Jacobi);
   cg.SetPreconditioner(diagonal); cg.SetOperator(spatial.Mass());
   cg.SetRelTol(1e-13); cg.SetAbsTol(0.0); cg.SetMaxIter(100);
   cg.SetPrintLevel(-1);
   cg.Mult(r,du);
   MFEM_VERIFY(cg.GetConverged(), "Mass solve failed.");
}
void FlowOperator::ImplicitSolve(real_t a, const Vector &base, Vector &state)
{
   MFEM_VERIFY(ImplicitVarTypeIsState(),
               "Implicit solves require stage-state mode.");
   spatial.SetTime(GetTime());
   stage.Configure(a,base);
   // SDIRK may leave a slope in its output workspace, so retain our own last
   // admissible stage for extrapolated bases outside the physical state set.
   if (spatial.Admissible(base)) { state = base; }
   else
   {
      MFEM_VERIFY(last_stage.Size() == height && spatial.Admissible(last_stage),
                  "No admissible stage initial guess.");
      state = last_stage;
   }
   LinearSolver linear(spatial,stage,options,stats,factory,external_solver);
   FlowNewton newton(spatial);
   newton.iterative_mode = true;
   newton.SetOperator(stage); newton.SetSolver(linear);
   newton.SetRelTol(options.newton_rtol); newton.SetAbsTol(options.newton_atol);
   newton.SetMaxIter(options.newton_max); newton.SetPrintLevel(-1);
   Vector zero;
   const int initial_linear_iterations = stats.linear_iterations;
   newton.Mult(zero,state);
   MFEM_VERIFY(newton.GetConverged(),
               "Newton failed at t=" << GetTime() << ", stage coefficient=" << a
               << ", iterations=" << newton.GetNumIterations()
               << ", residual=" << newton.GetFinalNorm()
               << " (including possible backtracking exhaustion).");
   stats.newton_iterations += newton.GetNumIterations();
   stats.residual = newton.GetFinalNorm(); stats.stages++;
   step_stage++;
   if (options.verbose)
   {
      int rank; MPI_Comm_rank(spatial.Space().GetComm(),&rank);
      if (rank == 0)
      {
         const int iterations = newton.GetNumIterations();
         const real_t average = iterations == 0 ? 0.0 :
                                real_t(stats.linear_iterations-initial_linear_iterations)/iterations;
         std::ostringstream line;
         line << "  Stage " << std::setw(2) << step_stage
              << "  Newton iterations = " << std::setw(4) << iterations
              << "  avg GMRES iterations = " << std::fixed << std::setprecision(2)
              << std::setw(8) << average;
         mfem::out << line.str() << std::endl;
      }
   }
   last_stage = state;
}
}
