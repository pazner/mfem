// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#include "flow.hpp"
#include <cmath>
#include <iostream>

using namespace mfem;
using namespace compressible;

namespace
{
int failures = 0;
void Check(bool condition, const char *message)
{
   int ok = condition, all_ok;
   MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
   if (!all_ok)
   {
      failures++;
      if (Mpi::Root()) { std::cerr << "FAIL: " << message << '\n'; }
   }
}
real_t Norm(const Vector &x)
{
   real_t a = x*x, b;
   MPI_Allreduce(&a,&b,1,MPITypeMap<real_t>::mpi_type,MPI_SUM,MPI_COMM_WORLD);
   return std::sqrt(b);
}
void CheckGradient(Operator &op, const Vector &u, const char *name)
{
   Vector v(u.Size()), rp(u.Size()), rm(u.Size()), jv(u.Size()), up(u), um(u);
   for (int i = 0; i < v.Size(); i++) { v(i) = std::sin(.7*(i+1)+Mpi::WorldRank()); }
   op.GetGradient(u).Mult(v,jv);
   const real_t eps = 1e-6;
   up.Add(eps,v); um.Add(-eps,v);
   op.Mult(up,rp); op.Mult(um,rm);
   rp -= rm; rp /= 2*eps; rp -= jv;
   const real_t relative = Norm(rp)/std::max(1.0,Norm(jv));
   if (Mpi::Root()) { std::cout << name << " relative error " << relative << '\n'; }
   Check(relative < 2e-7,name);
}
void LocalPhysics()
{
   Gas gas; gas.viscosity = .03;
   Vector u(4), du(4); u(0) = 1.2; u(1) = .7; u(2) = -.2; u(3) = 3.1;
   DenseMatrix q(4,2), dq(4,2), df, fp, fm;
   for (int i = 0; i < 4; i++)
   {
      du(i) = .1*(i+1);
      for (int d = 0; d < 2; d++) { q(i,d) = .2*std::sin(i+2*d); dq(i,d) = .3*std::cos(i+d); }
   }
   ViscousDerivative(gas,u,q,du,dq,df);
   Vector up(u), um(u); DenseMatrix qp(q), qm(q);
   up.Add(1e-6,du); um.Add(-1e-6,du); qp.Add(1e-6,dq); qm.Add(-1e-6,dq);
   ViscousFlux(gas,up,qp,fp); ViscousFlux(gas,um,qm,fm);
   fp -= fm; fp *= .5e6; fp -= df;
   Check(fp.MaxMaxNorm() < 1e-9,"analytic viscous differential");

   Mesh mesh = Mesh::MakeCartesian2D(2,1,Element::QUADRILATERAL,true);
   FaceElementTransformations *face = nullptr;
   for (int i = 0; i < mesh.GetNumFaces(); i++)
   { auto *tr = mesh.GetFaceElementTransformations(i); if (tr->Elem2No >= 0) { face = tr; break; } }
   face->SetAllIntPoints(&Geometries.GetCenter(Geometry::SEGMENT));
   EulerFlux flux(2,gas.gamma); EulerRusanovFlux numerical(flux,gas);
   Vector normal(2), left(u), right(u), fplus(4), fminus(4), f(4), swapped(4);
   right(0) += .2; right(1) -= .1; right(3) += .3;
   for (real_t sign : {-1.0,1.0})
   {
      normal(0) = sign*1.7; normal(1) = sign*.4;
      for (int side : {1,2})
      {
         DenseMatrix jac; numerical.Grad(side,left,right,normal,*face,jac);
         Vector &state = side == 1 ? left : right;
         for (int j = 0; j < 4; j++)
         {
            state(j) += 1e-6; numerical.Eval(left,right,normal,*face,fplus);
            state(j) -= 2e-6; numerical.Eval(left,right,normal,*face,fminus);
            state(j) += 1e-6;
            for (int i = 0; i < 4; i++)
            { Check(std::abs(jac(i,j)-(fplus(i)-fminus(i))/2e-6) < 2e-8,"exact Rusanov derivative"); }
         }
      }
      numerical.Eval(left,right,normal,*face,f);
      normal *= -1;
      numerical.Eval(right,left,normal,*face,swapped); swapped += f;
      Check(swapped.Normlinf() < 1e-14,"flux orientation");
   }
   // Tied speeds with nonzero jumps: symmetric central differences select the
   // averaged generalized derivative. Include the zero velocity norm case.
   for (bool zero_velocity : {false,true})
   {
      left = u;
      if (zero_velocity) { left(1) = left(2) = 0.0; }
      right = left; right *= 2.0;
      DenseMatrix jac;
      for (int side : {1,2})
      {
         numerical.Grad(side,left,right,normal,*face,jac);
         Vector &state = side == 1 ? left : right;
         for (int j = 0; j < 4; j++)
         {
            state(j) += 1e-7; numerical.Eval(left,right,normal,*face,fplus);
            state(j) -= 2e-7; numerical.Eval(left,right,normal,*face,fminus);
            state(j) += 1e-7;
            for (int i = 0; i < 4; i++)
            { Check(std::abs(jac(i,j)-(fplus(i)-fminus(i))/2e-7) < 2e-6,"Rusanov generalized derivative"); }
         }
      }
   }
}
void ManufacturedSource()
{
   Gas gas; gas.gamma = 1.67; gas.gas_constant = .8;
   gas.viscosity = .03; gas.prandtl = .9;
   Problem problem(gas,"manufactured");
   Mesh mesh = Mesh::MakeCartesian2D(1,1,Element::QUADRILATERAL,true);
   auto &tr = *mesh.GetElementTransformation(0);
   tr.SetIntPoint(&Geometries.GetCenter(Geometry::SQUARE));
   EulerFlux euler(2,gas.gamma);
   Vector x(2), u(4), up(4), um(4), approx(4), source(4);
   x(0) = .37; x(1) = -.42;
   const real_t t = .17, h = 1e-4;
   problem.State(x,t+h,up); problem.State(x,t-h,um);
   subtract(up,um,approx); approx /= 2*h;
   auto total_flux = [&](Vector point, DenseMatrix &f)
   {
      DenseMatrix grad(4,2), viscous;
      problem.State(point,t,u);
      for (int d = 0; d < 2; d++)
      {
         point(d) += h; problem.State(point,t,up);
         point(d) -= 2*h; problem.State(point,t,um); point(d) += h;
         for (int a = 0; a < 4; a++) { grad(a,d) = (up(a)-um(a))/(2*h); }
      }
      euler.ComputeFlux(u,tr,f);
      ViscousFlux(gas,u,grad,viscous); f -= viscous;
   };
   DenseMatrix fp(4,2), fm(4,2);
   for (int d = 0; d < 2; d++)
   {
      x(d) += h; total_flux(x,fp);
      x(d) -= 2*h; total_flux(x,fm); x(d) += h;
      for (int a = 0; a < 4; a++) { approx(a) += (fp(a,d)-fm(a,d))/(2*h); }
   }
   problem.Source(x,t,source); approx -= source;
   Check(approx.Normlinf() < 1e-7,"independent manufactured source");
   gas.viscosity = 0;
   DenseMatrix q(4,2), f, df; q = .3;
   Vector du(4); du = .2;
   ViscousFlux(gas,u,q,f); ViscousDerivative(gas,u,q,du,q,df);
   Check(f.MaxMaxNorm() == 0 &&
         df.MaxMaxNorm() == 0,"zero-viscosity flux and Jacobian");
}

void Parallel(bool triangles, real_t viscosity)
{
   Mesh mesh = PeriodicMesh(3,triangles);
   ParMesh pm(MPI_COMM_WORLD,mesh);
   DG_FECollection fec(2,2);
   ParFiniteElementSpace fes(&pm,&fec,4,Ordering::byNODES);
   Gas gas; gas.viscosity = viscosity;
   Problem problem(gas,"manufactured"), constant(gas,"constant");
   SpatialOperator spatial(fes,gas,problem), constant_op(fes,gas,constant);
   ParGridFunction gf(&fes);
   VectorFunctionCoefficient coef(4,[&](const Vector &x, Vector &u) { problem.State(x,.2,u); });
   gf.ProjectCoefficient(coef);
   Vector u; gf.GetTrueDofs(u); spatial.SetTime(.2);
   Check(spatial.Admissible(u),"projected manufactured state admissible");
   CheckGradient(spatial,u,"spatial Jacobian");
   SolverStats stats; StageOperator stage(spatial,stats); stage.Configure(.037,u);
   CheckGradient(stage,u,"shifted Jacobian");
   Vector v(u.Size()), a(u.Size()), b(u.Size()), c(u.Size()); v = 1.0;
   stage.GetGradient(u).Mult(v,a); spatial.GetGradient(u).Mult(v,b);
   spatial.Mass().Mult(v,c); a -= b; a.Add(-1/.037,c);
   Check(Norm(a) < 1e-10,"shifted matrix equals spatial Jacobian plus mass shift");
   if (!triangles && viscosity > 0)
   {
      SolverOptions opts;
      FlowOperator custom(spatial,opts), supplied(spatial,opts);
      int factory_calls = 0;
      custom.SetPreconditionerFactory([&](const PreconditionerContext &context)
      {
         factory_calls++;
         Check(context.mass_shift > 0 && context.state.Size() == u.Size(),
               "preconditioner context");
         // Requesting a spatial Jacobian must not invalidate the stage matrix.
         context.spatial.GetGradient(context.state);
         return std::unique_ptr<Solver>(new HypreDiagScale);
      });
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetRelTol(1e-10); gmres.SetAbsTol(0); gmres.SetMaxIter(300);
      gmres.SetKDim(100); gmres.SetPrintLevel(-1);
      supplied.SetLinearSolver(gmres);
      Vector first(u.Size()), second(u.Size());
      custom.SetTime(.21); supplied.SetTime(.21);
      custom.ImplicitSolve(.01,u,first); supplied.ImplicitSolve(.01,u,second);
      first -= second;
      Check(factory_calls > 0 &&
            Norm(first) < 1e-9,"custom preconditioner and supplied linear solver");
   }
   VectorFunctionCoefficient uniform(4,[&](const Vector &x, Vector &value) { constant.State(x,0,value); });
   gf.ProjectCoefficient(uniform); gf.GetTrueDofs(u);
   constant_op.Mult(u,a); Check(Norm(a) < 1e-12,"constant-state residual");
   Vector bad(u);
   if (Mpi::Root()) { bad(0) = -100.0; }
   Check(!constant_op.Admissible(bad),"collective admissibility rejection");
   SolverOptions options;
   FlowOperator flow(constant_op,options);
   auto ode = ODESolver::SelectImplicit(23); ode->Init(flow);
   real_t time = 0, dt = .1; Vector old(u);
   ode->Step(u,time,dt); u -= old;
   Check(Norm(u) < 1e-11,"implicit constant-state preservation");
}
}
int main(int argc, char *argv[])
{
   Mpi::Init(argc,argv); Hypre::Init();
   LocalPhysics();
   ManufacturedSource();
   for (bool triangles : {false,true})
      for (real_t mu : {0.0,.03}) { Parallel(triangles,mu); }
   if (Mpi::Root()) { std::cout << "Compressible checks: " << failures << " failures\n"; }
   return failures ? 1 : 0;
}
