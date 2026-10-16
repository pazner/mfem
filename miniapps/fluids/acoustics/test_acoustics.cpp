// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#include "acoustics.hpp"
#include <cmath>
#include <iostream>

using namespace mfem;
using namespace acoustics;
namespace
{
int failures = 0;
void Check(bool condition, const char *message)
{
   int local = condition, global;
   MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
   if (!global)
   {
      failures++;
      if (Mpi::Root())
      {
         std::cerr << "FAIL: " << message << '\n';
      }
   }
}
real_t Norm(const Vector &x)
{
   return GlobalNorm(x,MPI_COMM_WORLD);
}
void Physics()
{
   Medium medium;
   medium.density = 1.7;
   medium.speed = 2.3;
   AcousticFlux flux(medium);
   RusanovFlux numerical(flux);
   Mesh mesh = Mesh::MakeCartesian2D(2,1,Element::QUADRILATERAL,true);
   auto &tr = *mesh.GetElementTransformation(0);
   tr.SetIntPoint(&Geometries.GetCenter(Geometry::SQUARE));
   FaceElementTransformations *face = nullptr;
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      auto *candidate = mesh.GetFaceElementTransformations(f);
      if (candidate->Elem2No >= 0)
      {
         face = candidate;
         break;
      }
   }
   face->SetAllIntPoints(&Geometries.GetCenter(Geometry::SEGMENT));
   Vector left(3), right(3), direction(3);
   left(0) = -.6;
   left(1) = .8;
   left(2) = -.3;
   right(0) = .9;
   right(1) = -.4;
   right(2) = .2;
   direction(0) = .7;
   direction(1) = -.2;
   direction(2) = .3;
   DenseTensor jacobian;
   flux.ComputeFluxJacobian(left,tr,jacobian);
   DenseMatrix fp, fm;
   const real_t eps = 1e-6;
   Vector plus(left), minus(left);
   plus.Add(eps,direction);
   minus.Add(-eps,direction);
   flux.ComputeFlux(plus,tr,fp);
   flux.ComputeFlux(minus,tr,fm);
   fp -= fm;
   fp *= .5/eps;
   real_t derivative_error = 0;
   for (int d = 0; d < 2; d++)
      for (int i = 0; i < 3; i++)
      {
         real_t expected = 0;
         for (int j = 0; j < 3; j++)
         {
            expected += jacobian(i,j,d)*direction(j);
         }
         derivative_error = std::max(derivative_error,std::abs(fp(i,d)-expected));
      }
   Check(derivative_error < 2e-9,"physical flux derivative");
   for (real_t sign :
        {
           real_t(1),real_t(-1)
        })
   {
      Vector normal(2), f(3), reversed(3), a(3), b(3), jv(3);
      normal(0) = sign*1.2;
      normal(1) = sign*1.6;
      Check(numerical.Eval(left,right,normal,*face,f) == medium.speed,
            "constant characteristic speed");
      for (int side :
           {
              1,2
           })
      {
         Vector lp(left), lm(left), rp(right), rm(right);
         if (side == 1)
         {
            lp.Add(eps,direction);
            lm.Add(-eps,direction);
         }
         else
         {
            rp.Add(eps,direction);
            rm.Add(-eps,direction);
         }
         numerical.Eval(lp,rp,normal,*face,a);
         numerical.Eval(lm,rm,normal,*face,b);
         DenseMatrix j;
         numerical.Grad(side,left,right,normal,*face,j);
         j.Mult(direction,jv);
         a -= b;
         a *= .5/eps;
         a -= jv;
         Check(a.Normlinf() < 2e-9,
               "numerical flux derivative on scaled/reversed normal");
      }
      DenseMatrix physical;
      flux.ComputeFlux(left,tr,physical);
      physical.Mult(normal,a);
      numerical.Eval(left,left,normal,*face,b);
      a -= b;
      Check(a.Normlinf() < 1e-14,"consistent numerical flux");
      normal *= -1;
      numerical.Eval(right,left,normal,*face,reversed);
      reversed += f;
      Check(reversed.Normlinf() < 1e-14,"conservative orientation reversal");
   }
   for (const std::string name : {"wave", "multimode"})
   {
      // Independently check the analytic wave satisfies both acoustic equations.
      Problem problem(medium,name);
      Vector x(2), u(3), ut(3), div(3), xp(2), xm(2);
      x(0) = .3;
      x(1) = -.7;
      problem.State(x,.21+eps,plus);
      problem.State(x,.21-eps,minus);
      ut = plus;
      ut -= minus;
      ut *= .5/eps;
      div = 0.0;
      for (int d = 0; d < 2; d++)
      {
         xp = x;
         xm = x;
         xp(d) += eps;
         xm(d) -= eps;
         problem.State(xp,.21,plus);
         problem.State(xm,.21,minus);
         flux.ComputeFlux(plus,tr,fp);
         flux.ComputeFlux(minus,tr,fm);
         for (int i = 0; i < 3; i++)
         {
            div(i) += (fp(i,d)-fm(i,d))*.5/eps;
         }
      }
      ut += div;
      Check(ut.Normlinf() < 2e-8,"exact acoustic problem satisfies PDE");
      problem.State(x,.21,u);
      for (int d = 0; d < 2; d++)
      {
         xp = x;
         xp(d) += 4;
         problem.State(xp,.21,plus);
         plus -= u;
         Check(plus.Normlinf() < 1e-13,"exact problem is doubly periodic");
      }
   }
}
void Parallel(bool triangles)
{
   Mesh mesh = PeriodicMesh(3,triangles);
   ParMesh pm(MPI_COMM_WORLD,mesh);
   DG_FECollection fec(2,2);
   ParFiniteElementSpace fes(&pm,&fec,3,Ordering::byNODES);
   Medium medium;
   medium.density = 1.3;
   medium.speed = .8;
   SpatialOperator spatial(fes,medium);
   Vector u(fes.GetTrueVSize()), a(u.Size()), b(u.Size()), c(u.Size());
   for (int i = 0; i < u.Size(); i++)
   {
      u(i) = std::sin(.7*(i+1)+Mpi::WorldRank());
   }
   spatial.Mult(u,a);
   spatial.FormMult(u,b);
   a -= b;
   Check(Norm(a) < 1e-11,
         "assembled spatial matrix agrees with form including MPI faces");
   spatial.Mult(u,a);
   spatial.EnergyWeights(a);
   real_t local = u*a, dissipation;
   MPI_Allreduce(&local,&dissipation,1,MPITypeMap<real_t>::mpi_type,MPI_SUM,
                 MPI_COMM_WORLD);
   Check(dissipation >= -1e-12,"semidiscrete acoustic energy dissipation");
   SolverOptions opts;
   GMRESSolver external(MPI_COMM_WORLD);
   AcousticOperator flow(spatial,opts);
   int factory_calls = 0;
   flow.SetPreconditionerFactory([&](const PreconditionerContext &context)
   {
      factory_calls++;
      context.shifted_matrix.Mult(u,a);
      context.spatial.GetGradient(u).Mult(u,b);
      context.spatial.Mass().Mult(u,c);
      a -= b;
      a.Add(-context.mass_shift,c);
      Check(Norm(a) < 1e-10,"shifted matrix equals K plus mass shift");
      return std::unique_ptr<Solver>(new HypreDiagScale);
   });
   Vector state(u.Size());
   auto solve = [&](real_t dt)
   {
      flow.ImplicitSolve(dt,u,state);
      spatial.Mult(state,a);
      b = state;
      b -= u;
      spatial.Mass().Mult(b,c);
      a.Add(1/dt,c);
      Check(Norm(a) < 1e-9,"implicit stage satisfies original equation");
   };
   solve(.01);
   solve(.01);
   Check(factory_calls == 1 &&
         flow.Stats().setups == 1,"reuse equal stage coefficients");
   solve(.006);
   Check(factory_calls == 2 &&
         flow.Stats().setups == 2,"rebuild changed stage coefficient");
   flow.SetPreconditionerFactory([&](const PreconditionerContext &)
   {
      factory_calls++;
      return std::unique_ptr<Solver>(new HypreDiagScale);
   });
   solve(.006);
   Check(factory_calls == 3,"factory replacement invalidates setup");
   external.SetRelTol(1e-12);
   external.SetAbsTol(0);
   external.SetMaxIter(300);
   external.SetKDim(100);
   external.SetPrintLevel(-1);
   flow.SetLinearSolver(external);
   solve(.006);
   Check(flow.Stats().setups == 4 &&
         factory_calls == 3,"caller-owned solver replaces built-in solver");
   flow.SetLinearSolver(external);
   solve(.006);
   Check(flow.Stats().setups == 5,"linear solver setter invalidates setup");
   // Verify that every block contains all components of exactly one element.
   // Apply an independent D or D+L in the original component ordering.
   ConstantCoefficient shift(100.0);
   VectorMassIntegrator mass(shift);
   mass.SetVDim(3);
   ParNonlinearForm stage(&fes);
   spatial.AddIntegrators(stage);
   stage.AddDomainIntegrator(&mass);
   auto &matrix = dynamic_cast<HypreParMatrix &>(stage.GetGradient(u));
   SparseMatrix processor_matrix;
   matrix.GetDiag(processor_matrix);
   Array<int> element_of_dof(u.Size()), dofs;
   bool complete_blocks = true;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      fes.GetElementVDofs(e,dofs);
      complete_blocks = complete_blocks && dofs.Size() == 3*fes.GetFE(e)->GetDof();
      for (int dof : dofs)
      {
         element_of_dof[dof] = e;
      }
   }
   Check(complete_blocks,"element block includes all three components");
   for (const std::string type :
        {"block-jacobi", "block-gs", "block-ilu"
        })
   {
      ElementBlockPreconditioner preconditioner(fes,type);
      preconditioner.SetOperator(matrix);
      preconditioner.Mult(u,a);
      if (type != "block-ilu")
      {
         b = 0.0;
         for (int i = 0; i < processor_matrix.Height(); i++)
         {
            for (int k = processor_matrix.GetI()[i]; k < processor_matrix.GetI()[i+1]; k++)
            {
               const int j = processor_matrix.GetJ()[k];
               if (element_of_dof[i] == element_of_dof[j] ||
                   (type == "block-gs" && element_of_dof[j] < element_of_dof[i]))
               {
                  b(i) += processor_matrix.GetData()[k]*a(j);
               }
            }
         }
         b -= u;
         Check(Norm(b) < 1e-10,"element block inverse in original component ordering");
      }
      // Every element is isolated here: block ILU must invert each complete
      // element block exactly, despite noncontiguous component storage.
      SparseMatrix element_diagonal(processor_matrix.Height());
      for (int i = 0; i < processor_matrix.Height(); i++)
      {
         for (int k = processor_matrix.GetI()[i]; k < processor_matrix.GetI()[i+1]; k++)
         {
            const int j = processor_matrix.GetJ()[k];
            if (element_of_dof[i] == element_of_dof[j])
            {
               element_diagonal.Add(i,j,processor_matrix.GetData()[k]);
            }
         }
      }
      element_diagonal.Finalize();
      HypreParMatrix isolated(MPI_COMM_WORLD,fes.GlobalTrueVSize(),
                              fes.GetTrueDofOffsets(),&element_diagonal);
      preconditioner.SetOperator(isolated);
      preconditioner.Mult(u,a);
      isolated.Mult(a,b);
      b -= u;
      Check(Norm(b) < 1e-10,"exact solve of uncoupled full element blocks");

      SolverOptions selected;
      selected.preconditioner = type;
      AcousticOperator evolution(spatial,selected);
      evolution.ImplicitSolve(.01,u,a);
      spatial.Mult(a,b);
      a -= u;
      spatial.Mass().Mult(a,c);
      b.Add(100.0,c);
      Check(Norm(b) < 1e-9,"implicit solve with each element preconditioner");
   }

   Problem constant(medium,"constant");
   ParGridFunction gf(&fes);
   VectorFunctionCoefficient coefficient(3,[&](const Vector &x, Vector &value)
   {
      constant.State(x,0,value);
   });
   gf.ProjectCoefficient(coefficient);
   gf.GetTrueDofs(u);
   spatial.Mult(u,a);
   Check(Norm(a) < 1e-12,"constant-state spatial residual");
   for (int method :
        {
           4,21,22,23
        })
   {
      AcousticOperator evolution(spatial,opts);
      auto ode = method == 4 ? ODESolver::SelectExplicit(4) :
                 ODESolver::SelectImplicit(method);
      if (method == 4)
      {
         evolution.SetExplicitMode();
      }
      ode->Init(evolution);
      Vector value(u);
      real_t time = 0, dt = .01;
      ode->Step(value,time,dt);
      value -= u;
      Check(Norm(value) < 1e-10,"constant preservation for every time integrator");
   }
   // Direct algebraic solves share the stage setup but do not multiply by M/a.
   Vector rhs, repeated;
   RandomRightHandSide(fes,rhs);
   RandomRightHandSide(fes,repeated);
   repeated -= rhs;
   Check(Norm(repeated) == 0 && std::abs(Norm(rhs)-1) < 1e-14,
         "random RHS is reproducible and normalized");
   AcousticOperator algebraic(spatial,opts);
   for (real_t dt : {real_t(.01), real_t(.01), real_t(.006)})
   {
      algebraic.SolveSystem(dt,rhs,state);
      spatial.Mult(state,a);
      spatial.Mass().Mult(state,b);
      a.Add(1/dt,b);
      a -= rhs;
      Check(Norm(a) < 1e-10,"direct solve satisfies shifted system with raw RHS");
   }
   Check(algebraic.Stats().stages == 3 && algebraic.Stats().setups == 2,
         "direct solve reuses and rebuilds the cached setup");

   // Pressure perturbations need not be positive, and zero RHS must converge.
   u = 0.0;
   AcousticOperator zero(spatial,opts);
   zero.ImplicitSolve(.01,u,state);
   Check(Norm(state) == 0,"zero-state implicit solve");
}
}
int main(int argc, char *argv[])
{
   Mpi::Init(argc,argv);
   Hypre::Init();
   Physics();
   Parallel(false);
   Parallel(true);
   if (Mpi::Root())
   {
      std::cout << "Acoustics checks: " << failures << " failures\n";
   }
   return failures ? 1 : 0;
}
