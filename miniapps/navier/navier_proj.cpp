// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// Navier MMS example
//
// A manufactured solution is defined as
//
// u = [pi * sin(t) * sin(pi * x)^2 * sin(2 * pi * y),
//      -(pi * sin(t) * sin(2 * pi * x)) * sin(pi * y)^2].
//
// p = cos(pi * x) * sin(t) * sin(pi * y)
//
// The solution is used to compute the symbolic forcing term (right hand side),
// of the equation. Then the numerical solution is computed and compared to the
// exact manufactured solution to determine the error.

#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

constexpr real_t pi = M_PI;

struct s_NavierContext
{
   int ser_ref_levels = 1;
   int order = 3;
   real_t kinvis = 1.0;
   real_t t_final = 1.0;
   real_t dt = 0.25e-4;
   bool ni = false;
} ctx;

void vel(const Vector &xvec, real_t t, Vector &u)
{
   real_t x = xvec(0);
   real_t y = xvec(1);

   u(0) = pi*sin(t) * sin(pi*x)*sin(pi*x) * sin(2.0*pi*y);
   u(1) = -pi*sin(t) * sin(2.0*pi*x) * sin(pi*y)*sin(pi*y);
}

real_t p(const Vector &xvec, real_t t)
{
   real_t x = xvec(0);
   real_t y = xvec(1);

   return sin(t) * cos(pi*x) * sin(pi*y);
}

void accel(const Vector &xvec, real_t t, Vector &fvec)
{
   real_t x = xvec(0);
   real_t y = xvec(1);

   fvec[0] = -(pi*sin(t)*sin(pi*x)*sin(pi*y)) +
             pi*(2*pow(pi,2)*(1 - 2*cos(2*pi*x))*sin(t) +
                 cos(t)*pow(sin(pi*x),2))*sin(2*pi*y);

   fvec[1] = pi*(cos(pi*x)*cos(pi*y)*sin(t) +
                 sin(2*pi*x)*(2*pow(pi,2)*(-1 + 2*cos(2*pi*y))*sin(t) -
                              cos(t)*pow(sin(pi*y),2)));

   // u(0) = pi * sin(t) * sin(pi * xi) * sin(pi * yi)
   //        * (-1.0
   //           + 2.0 * pow(pi, 2.0) * sin(t) * sin(pi * xi)
   //           * sin(2.0 * pi * xi) * sin(pi * yi))
   //        + pi
   //        * (2.0 * ctx.kinvis * pow(pi, 2.0)
   //           * (1.0 - 2.0 * cos(2.0 * pi * xi)) * sin(t)
   //           + cos(t) * pow(sin(pi * xi), 2.0))
   //        * sin(2.0 * pi * yi);

   // u(1) = pi * cos(pi * yi) * sin(t)
   //        * (cos(pi * xi)
   //           + 2.0 * ctx.kinvis * pow(pi, 2.0) * cos(pi * yi)
   //           * sin(2.0 * pi * xi))
   //        - pi * (cos(t) + 6.0 * ctx.kinvis * pow(pi, 2.0) * sin(t))
   //        * sin(2.0 * pi * xi) * pow(sin(pi * yi), 2.0)
   //        + 4.0 * pow(pi, 3.0) * cos(pi * yi) * pow(sin(t), 2.0)
   //        * pow(sin(pi * xi), 2.0) * pow(sin(pi * yi), 3.0);
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&ctx.order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&ctx.ni, "-ni", "--enable-ni", "-no-ni", "--disable-ni",
                  "Enable numerical integration rules.");
   args.ParseCheck();

   Mesh *mesh = new Mesh("../../data/inline-quad.mesh");
   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   *nodes *= 2.0;
   *nodes -= 1.0;

   for (int i = 0; i < ctx.ser_ref_levels; ++i)
   {
      mesh->UniformRefinement();
   }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Create the flow solver.
   NavierSolver naviersolver(pmesh, ctx.order, ctx.kinvis);
   naviersolver.EnableNI(ctx.ni);

   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel);
   FunctionCoefficient p_excoeff(p);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 1;
   naviersolver.AddVelDirichletBC(vel, attr);

   Array<int> domain_attr(pmesh->attributes.Max());
   domain_attr = 1;
   naviersolver.AddAccelTerm(accel, domain_attr);

   real_t t = 0.75;
   real_t dt = ctx.dt;
   real_t t_final = ctx.t_final;
   bool last_step = false;

   u_excoeff.SetTime(t);
   p_excoeff.SetTime(t);

   ParGridFunction *u_gf = naviersolver.GetCurrentVelocity();
   ParGridFunction *p_gf = naviersolver.GetCurrentPressure();

   ParGridFunction u_ex(naviersolver.GetCurrentVelocity()->ParFESpace());
   ParGridFunction p_ex(naviersolver.GetCurrentPressure()->ParFESpace());

   u_ex.ProjectCoefficient(u_excoeff);
   p_ex.ProjectCoefficient(p_excoeff);

   u_gf->ProjectCoefficient(u_excoeff);
   p_gf->ProjectCoefficient(p_excoeff);

   naviersolver.Setup(dt);

   if (Mpi::Root())
   {
      printf("\n");
      printf("%-11s  %-11s   %-11s    %-11s\n",
             "Time", "dt", "Vel. Err.", "Pres. Err.");
      printf("-----------------------------------------------------\n");
   }

   ParaViewDataCollection pv("NavierProj", pmesh);
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   pv.SetLevelsOfDetail(ctx.order);
   pv.RegisterField("u", u_gf);
   pv.RegisterField("p", p_gf);
   pv.RegisterField("u exact", &u_ex);
   pv.RegisterField("p exact", &p_ex);
   pv.SetCycle(0);
   pv.SetTime(0);
   pv.Save();

   while (!last_step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      u_excoeff.SetTime(t);
      p_excoeff.SetTime(t);

      u_gf->ProjectCoefficient(u_excoeff);

      // naviersolver.Step(t, dt, step);
      naviersolver.StepFirstOrder(t, dt);

      // Compare against exact solution of velocity and pressure.
      u_excoeff.SetTime(t);
      p_excoeff.SetTime(t);

      u_gf->ProjectCoefficient(u_excoeff);

      const real_t err_u = u_gf->ComputeL2Error(u_excoeff);
      const real_t err_p = p_gf->ComputeL2Error(p_excoeff);

      if (Mpi::Root())
      {
         printf("%-11.3e  %-11.3e   %.5e    %.5e\n", t, dt, err_u, err_p);
         fflush(stdout);
      }

      u_ex.ProjectCoefficient(u_excoeff);
      p_ex.ProjectCoefficient(p_excoeff);

      pv.SetTime(t);
      pv.SetCycle(pv.GetCycle() + 1);
      pv.Save();
   }

   delete pmesh;

   return 0;
}
