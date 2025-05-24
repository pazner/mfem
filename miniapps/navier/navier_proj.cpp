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
bool nonlinear = false;

void vel(const Vector &xvec, real_t t, Vector &u)
{
   real_t x = xvec[0];
   real_t y = xvec[1];

   u[0] = pi*sin(t) * sin(pi*x)*sin(pi*x) * sin(2.0*pi*y);
   u[1] = -pi*sin(t) * sin(2.0*pi*x) * sin(pi*y)*sin(pi*y);
}

real_t p(const Vector &xvec, real_t t)
{
   real_t x = xvec[0];
   real_t y = xvec[1];

   return sin(t) * cos(pi*x) * sin(pi*y);
}

void accel(const Vector &xvec, real_t t, Vector &fvec)
{
   real_t x = xvec[0];
   real_t y = xvec[1];

   if (nonlinear)
   {
      fvec[0] = pi * sin(t) * sin(pi * x) * sin(pi * y)
                * (-1.0
                   + 2.0 * pow(pi, 2.0) * sin(t) * sin(pi * x)
                   * sin(2.0 * pi * x) * sin(pi * y))
                + pi
                * (2.0 * pow(pi, 2.0)
                   * (1.0 - 2.0 * cos(2.0 * pi * x)) * sin(t)
                   + cos(t) * pow(sin(pi * x), 2.0))
                * sin(2.0 * pi * y);

      fvec[1] = pi * cos(pi * y) * sin(t)
                * (cos(pi * x)
                   + 2.0 * pow(pi, 2.0) * cos(pi * y)
                   * sin(2.0 * pi * x))
                - pi * (cos(t) + 6.0 * pow(pi, 2.0) * sin(t))
                * sin(2.0 * pi * x) * pow(sin(pi * y), 2.0)
                + 4.0 * pow(pi, 3.0) * cos(pi * y) * pow(sin(t), 2.0)
                * pow(sin(pi * x), 2.0) * pow(sin(pi * y), 3.0);
   }
   else
   {
      fvec[0] = -(pi*sin(t)*sin(pi*x)*sin(pi*y)) +
                pi*(2*pow(pi,2)*(1 - 2*cos(2*pi*x))*sin(t) +
                    cos(t)*pow(sin(pi*x),2))*sin(2*pi*y);

      fvec[1] = pi*(cos(pi*x)*cos(pi*y)*sin(t) +
                    sin(2*pi*x)*(2*pow(pi,2)*(-1 + 2*cos(2*pi*y))*sin(t) -
                                 cos(t)*pow(sin(pi*y),2)));
   }
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   // Number of times to refine the mesh (serial refinement levels)
   int ser_ref_levels = 1;
   int order = 3; // Degree of the velocity space
   real_t t_initial = 0.0;
   real_t t_final = 1.0;
   real_t dt = 0.25e-4;

   int time_order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&t_initial, "-ti", "--initial-time", "Initial time.");
   args.AddOption(&t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&nonlinear, "-nl", "--nonlinear", "-lin", "--linear",
                  "Include the nonlinear term?");
   args.AddOption(&time_order, "-to", "--time-order",
                  "Time integration order (1, 2, or 3).");
   args.ParseCheck();

   MFEM_VERIFY(time_order >= 1 &&
               time_order <= 3, "Time order must be 1, 2, or 3.");

   Mesh mesh("../../data/inline-quad.mesh");
   mesh.EnsureNodes();
   GridFunction *nodes = mesh.GetNodes();
   *nodes *= 2.0;
   *nodes -= 1.0;

   for (int i = 0; i < ser_ref_levels; ++i)
   {
      mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Create the flow solver.
   NavierSolver naviersolver(&pmesh, order, 1.0, nonlinear);

   VectorFunctionCoefficient u_excoeff(pmesh.Dimension(), vel);
   FunctionCoefficient p_excoeff(p);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh.bdr_attributes.Max());
   attr = 1;
   naviersolver.AddVelDirichletBC(vel, attr);

   Array<int> domain_attr(pmesh.attributes.Max());
   domain_attr = 1;
   naviersolver.AddAccelTerm(accel, domain_attr);

   real_t t = t_initial;
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

   ParaViewDataCollection pv("NavierProj", &pmesh);
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   pv.SetLevelsOfDetail(order);
   pv.RegisterField("u", u_gf);
   pv.RegisterField("p", p_gf);
   pv.RegisterField("u exact", &u_ex);
   pv.RegisterField("p exact", &p_ex);
   pv.SetCycle(0);
   pv.SetTime(0);
   pv.Save();

   naviersolver.StartUp(u_excoeff, p_excoeff, t, dt);
   int step = 2;

   while (!last_step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      u_excoeff.SetTime(t);
      p_excoeff.SetTime(t);

      if (time_order == 1)
      {
         naviersolver.StepFirstOrder(t, dt);
      }
      else if (time_order == 2)
      {
         naviersolver.StepIncremental(t, dt, step);
      }
      else
      {
         naviersolver.Step(t, dt, step);
      }

      ++step;

      // Compare against exact solution of velocity and pressure.
      u_excoeff.SetTime(t);
      p_excoeff.SetTime(t);

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

   return 0;
}
