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
// the paper
//
// T. A. Brunner, Development of a grey nonlinear thermal radiation diffusion
// verification problem (2006).

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

   // const int n_l2 = rad_diff.fes_l2.GetTrueVSize();
   // const int n_rt = rad_diff.fes_rt.GetTrueVSize();

   Vector u(rad_diff.Height());
   GridFunction u_e(&rad_diff.fes_l2, u, rad_diff.offsets[0]);
   GridFunction u_E(&rad_diff.fes_l2, u, rad_diff.offsets[1]);
   GridFunction u_F(&rad_diff.fes_rt, u, rad_diff.offsets[2]);

   FunctionCoefficient e_init_coeff(MMS::InitialMaterialEnergy);
   FunctionCoefficient E_init_coeff(MMS::InitialRadiationEnergy);

   u_e.ProjectCoefficient(e_init_coeff);
   u_E.ProjectCoefficient(E_init_coeff);
   u_F = 0.0;

   Vector k(rad_diff.Height());
   k = 0.0;

   std::cout << "\n\n === TESTING TIMESTEP ===\n\n";
   rad_diff.ImplicitSolve(rad_diff.dt, u, k);

   // std::cout << "\n\n === TESTING LINEAR SOLVER ===\n\n";

   // {
   //    // Test linear solver (Brunner-Nowack iteration)
   //    Vector r(rad_diff.Height());
   //    Vector x(rad_diff.Height());

   //    r.Randomize(1);
   //    x = 0.0;

   //    RadiationDiffusionLinearSolver linsolver(rad_diff);
   //    linsolver.Update(r);
   //    linsolver.Mult(r, x);
   // }

   // std::cout << "\n\n === TESTING NEWTON ===\n\n";
   // // return 0;

   // {
   //    struct MyOp : Operator
   //    {
   //       ParBilinearForm L_form;
   //       ParNonlinearForm H_form;
   //       std::unique_ptr<HypreParMatrix> L;
   //       mutable std::unique_ptr<HypreParMatrix> J;
   //       mutable Vector z;

   //       MyOp(RadiationDiffusionOperator &rd)
   //          : Operator(rd.fes_l2.GetTrueVSize()), L_form(&rd.fes_l2), H_form(&rd.fes_l2)
   //       {
   //          L_form.AddDomainIntegrator(new MassIntegrator);
   //          L_form.Assemble();
   //          L_form.Finalize();
   //          L.reset(L_form.ParallelAssemble());
   //          H_form.AddDomainIntegrator(new NonlinearEnergyIntegrator(rd));
   //       }

   //       void Mult(const Vector &x, Vector &y) const override
   //       {
   //          z.SetSize(y.Size());
   //          L->Mult(x, y);

   //          H_form.Mult(x, z);
   //          y.Add(1.0, z);
   //       }

   //       Operator &GetGradient(const Vector &x) const override
   //       {
   //          auto &dH = dynamic_cast<HypreParMatrix&>(H_form.GetGradient(x));
   //          J.reset(ParAdd(L.get(), &dH));
   //          return *J;
   //       }
   //    };

   //    MyOp op(rad_diff);

   //    // Test nonlinear solve and linearization
   //    ParFiniteElementSpace &fes_l2 = rad_diff.GetL2Space();

   //    ParNonlinearForm H_form(&fes_l2);
   //    H_form.AddDomainIntegrator(new NonlinearEnergyIntegrator(rad_diff));

   //    rad_diff.e_gf.ProjectCoefficient(e_init_coeff);
   //    // rad_diff.dt = 0.0;

   //    SerialDirectSolver direct_solver;
   //    NewtonSolver newton;
   //    newton.SetAbsTol(1e-9);
   //    newton.SetRelTol(1e-12);
   //    newton.SetMaxIter(100);
   //    newton.SetPrintLevel(IterativeSolver::PrintLevel().All());
   //    newton.SetSolver(direct_solver);
   //    newton.SetOperator(op);

   //    Vector b; // Treated as zero

   //    ParGridFunction x(&fes_l2), z(&fes_l2);
   //    z = 0.0;
   //    rad_diff.H_form.Mult(z, x);

   //    SerialDirectSolver L_solver(*rad_diff.L);
   //    L_solver.Mult(x, z);
   //    x.Set(-1.0, z);

   //    // op.Mult(x, z);
   //    // std::cout << z.Normlinf() << '\n';
   //    // return 0;

   //    // x = 0.0;

   //    newton.Mult(b, x);

   //    // Vector z(b.Size());
   //    // H_form.Mult(x, z);
   //    // z -= b;
   //    // std::cout << "Resnorm: " << z.Norml2() << '\n';
   //    // std::cout << "b norm:  " << b.Norml2() << '\n';
   //    // std::cout << "x norm:  " << x.Norml2() << '\n';
   // }

   // return 0;

   // {
   //    // Test nonlinear solve and linearization
   //    ParFiniteElementSpace &fes_l2 = rad_diff.GetL2Space();

   //    ParGridFunction b(&fes_l2);
   //    ParGridFunction x(&fes_l2);

   //    // rad_diff.e_gf.Randomize(1);
   //    rad_diff.dt = 1e-40;

   //    ParNonlinearForm H_form(&fes_l2);
   //    H_form.AddDomainIntegrator(new NonlinearEnergyIntegrator(rad_diff));

   //    // x.Randomize(5);
   //    // x *= 1e-2;
   //    // H_form.Mult(x, b);
   //    b = 0.0;

   //    std::cout << "\ndt: " << rad_diff.dt << "\n\n";
   //    std::cout << "\nb norm: " << b.Norml2() << "\n\n";

   //    // Use petrubation of solution as initial guess
   //    // Vector tmp(x.Size());
   //    // tmp.Randomize(1);
   //    // tmp -= 0.5;
   //    // tmp *= 1e-2;
   //    // tmp += 1.0;
   //    // x *= tmp;

   //    SerialDirectSolver direct_solver;
   //    NewtonSolver newton;
   //    newton.SetAbsTol(1e-12);
   //    newton.SetRelTol(1e-12);
   //    newton.SetMaxIter(100);
   //    newton.SetPrintLevel(IterativeSolver::PrintLevel().All());
   //    newton.SetSolver(direct_solver);
   //    newton.SetOperator(H_form);

   //    newton.Mult(b, x);

   //    Vector z(b.Size());
   //    H_form.Mult(x, z);
   //    z -= b;
   //    std::cout << "Resnorm: " << z.Norml2() << '\n';
   //    std::cout << "b norm:  " << b.Norml2() << '\n';
   //    std::cout << "x norm:  " << x.Norml2() << '\n';

   //    // {
   //    //    direct_solver.SetOperator(H_form.GetGradient(x));
   //    //    direct_solver.Mult(b, x);
   //    //    H_form.Mult(x, z);
   //    //    z -= b;
   //    //    std::cout << "Resnorm: " << z.Norml2() << '\n';
   //    // }

   //    // {
   //    //    ParBilinearForm m(&fes_l2);
   //    //    m.AddDomainIntegrator(new MassIntegrator);
   //    //    m.Assemble();
   //    //    m.Finalize();
   //    //    std::unique_ptr<HypreParMatrix> M(m.ParallelAssemble());
   //    //    M->Print("M.txt");
   //    //    SerialDirectSolver Minv(*M);

   //    //    x = 0.0;

   //    //    Minv.Mult(b, x);
   //    //    m.Mult(x, z);
   //    //    z -= b;
   //    //    std::cout << "Resnorm: " << z.Norml2() << '\n';
   //    // }

   // }

   return 0;
}
