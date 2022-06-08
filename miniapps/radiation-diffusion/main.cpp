// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
//                     ----------------------------------
//                     Radiation-Diffusion Solver Miniapp
//                     ----------------------------------
//
// This miniapp solves a simple radiation-diffusion test case as described in
// the paper:
//
//    T. A. Brunner, Development of a grey nonlinear thermal radiation diffusion
//    verification problem (2006). SAND2006-4030C.

#include "mfem.hpp"

#include "radiation_diffusion.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref = 0, int par_ref = 0)
{
   Mesh serial_mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   return mesh;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../../data/inline-quad.mesh";
   const char *device_config = "cpu";
   int ser_ref = 1;
   int par_ref = 1;
   int order = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");

   RadiationDiffusionOperator rad_diff(mesh, order);

   ParFiniteElementSpace &fes_l2 = rad_diff.GetL2Space();
   ParFiniteElementSpace &fes_rt = rad_diff.GetRTSpace();
   const Array<int> &offsets = rad_diff.GetOffsets();

   Vector u(rad_diff.Height());
   ParGridFunction u_e(&fes_l2, u, offsets[0]);
   ParGridFunction u_E(&fes_l2, u, offsets[1]);
   ParGridFunction u_F(&fes_rt, u, offsets[2]);

   FunctionCoefficient e_init_coeff(MMS::InitialMaterialEnergy);
   FunctionCoefficient E_init_coeff(MMS::InitialRadiationEnergy);

   u_e.ProjectCoefficient(e_init_coeff);
   u_E.ProjectCoefficient(E_init_coeff);
   u_F = 0.0;

   ParaViewDataCollection pv("RadiationDiffusion", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.RegisterField("e", &u_e);
   pv.RegisterField("E", &u_E);
   pv.RegisterField("F", &u_F);

   ParGridFunction u_e_exact(&fes_l2);
   ParGridFunction u_E_exact(&fes_l2);
   FunctionCoefficient e_exact_coeff(MMS::ExactMaterialEnergy);
   FunctionCoefficient E_exact_coeff(MMS::ExactRadiationEnergy);

   pv.RegisterField("e_exact", &u_e_exact);
   pv.RegisterField("E_exact", &u_E_exact);

   SDIRK33Solver ode;
   ode.Init(rad_diff);

   double dt;
   {
      double h_min, h_max, kappa_min, kappa_max;
      mesh.GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
      dt = h_min*0.05/MMS::tau;
   }

   double t = 0.0;
   const double tf = 0.1/MMS::tau;
   int i = 0;

   while (t < tf)
   {
      if (t + dt > tf) { dt = tf - t; }
      std::cout << "=== Step " << ++i << std::setprecision(2)
                << " t = " << t
                << " dt = " << dt
                << " ===\n" << std::endl;
      ode.Step(u, t, dt);

      e_exact_coeff.SetTime(t);
      E_exact_coeff.SetTime(t);
      u_e_exact.ProjectCoefficient(e_exact_coeff);
      u_E_exact.ProjectCoefficient(E_exact_coeff);

      pv.SetCycle(pv.GetCycle() + 1);
      pv.SetTime(t);
      pv.Save();
   }

   e_exact_coeff.SetTime(t);
   E_exact_coeff.SetTime(t);

   double e_error = u_e.ComputeL2Error(e_exact_coeff);
   double E_error = u_E.ComputeL2Error(E_exact_coeff);

   std::cout << '\n'
             << "Material energy error:  " << e_error << '\n'
             << "Radiation energy error: " << E_error << std::endl;

   return 0;
}
