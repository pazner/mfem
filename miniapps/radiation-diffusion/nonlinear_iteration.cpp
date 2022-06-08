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

#include "nonlinear_iteration.hpp"
#include "radiation_diffusion.hpp"

namespace mfem
{

NonlinearEnergyOperator::NonlinearEnergyOperator(
   RadiationDiffusionOperator &rad_diff_)
   : Operator(rad_diff_.offsets[2]),
     rad_diff(rad_diff_)
{ }

void NonlinearEnergyOperator::Mult(const Vector &x, Vector &y) const
{
   using namespace MMS;

   const double dt = rad_diff.dt;
   const int n_l2 = rad_diff.fes_l2.GetTrueVSize();
   const Array<int> &offsets = rad_diff.offsets;

   const Vector x_e(const_cast<Vector&>(x), offsets[0], n_l2);
   const Vector x_E(const_cast<Vector&>(x), offsets[1], n_l2);

   Vector y_e(y, offsets[0], n_l2);
   Vector y_E(y, offsets[1], n_l2);

   // Material energy mass term
   z.SetSize(n_l2);
   rad_diff.L->Mult(x_e, y_e); // Contribution to material energy
   y_e *= rho;

   // Material energy nonlinear term
   rad_diff.H_form.Mult(x_e, z);
   y_e += z; // Contribution to material energy
   y_E.Set(-1, z); // Contribution to radiation energy

   // Radiation energy mass term
   rad_diff.L->Mult(x_E, z);
   y_E.Add(1 + c*dt*sigma, z); // Contribution to radiation energy
   y_e.Add(-c*dt*sigma, z); // Contribution to material energy
}

Operator &NonlinearEnergyOperator::GetGradient(const Vector &x) const
{
   using namespace MMS;

   const Vector x_e(const_cast<Vector&>(x), 0, rad_diff.fes_l2.GetTrueVSize());

   Operator &dH = rad_diff.H_form.GetGradient(x_e);
   auto *dH_matrix = dynamic_cast<HypreParMatrix*>(&dH);

   MFEM_VERIFY(dH_matrix != nullptr, "");

   J00.reset(Add(rho, *rad_diff.L, 1.0, *dH_matrix));

   Array2D<HypreParMatrix*> eE_blocks(2,2);
   eE_blocks(0,0) = J00.get();
   eE_blocks(0,1) = rad_diff.L.get();
   eE_blocks(1,0) = dH_matrix;
   eE_blocks(1,1) = rad_diff.L.get();

   Array2D<double> eE_coeff(2,2);
   eE_coeff(0,0) = 1.0;
   eE_coeff(0,1) = -c*rad_diff.dt*sigma;
   eE_coeff(1,0) = -1.0;
   eE_coeff(1,1) = 1.0 + c*rad_diff.dt*sigma;

   J.reset(HypreParMatrixFromBlocks(eE_blocks, &eE_coeff));

   return *J;
}

BrunnerNowackIteration::BrunnerNowackIteration(
   RadiationDiffusionOperator &rad_diff_)
   : rad_diff(rad_diff_),
     N_eE(rad_diff),
     EF_solver(rad_diff)
{
   eE_solver.SetMaxIter(20);
   eE_solver.SetRelTol(1e-8);
   eE_solver.SetAbsTol(1e-8);
   eE_solver.SetOperator(N_eE);
   eE_solver.SetSolver(J_eE_solver);
   // eE_solver.SetPrintLevel(IterativeSolver::PrintLevel().All());
   eE_solver.SetPrintLevel(IterativeSolver::PrintLevel().None());
}

void BrunnerNowackIteration::ApplyFullOperator(const Vector &x, Vector &y) const
{
   using namespace MMS;

   const double dt = rad_diff.dt;
   const int n_l2 = rad_diff.fes_l2.GetTrueVSize();
   const int n_rt = rad_diff.fes_rt.GetTrueVSize();
   const Array<int> &offsets = rad_diff.offsets;

   const Vector x_e(const_cast<Vector&>(x), offsets[0], n_l2);
   const Vector x_E(const_cast<Vector&>(x), offsets[1], n_l2);
   const Vector x_F(const_cast<Vector&>(x), offsets[2], n_rt);

   Vector y_e(y, offsets[0], n_l2);
   Vector y_E(y, offsets[1], n_l2);
   Vector y_F(y, offsets[2], n_rt);

   // Material energy mass term
   z.SetSize(n_l2);
   rad_diff.L->Mult(x_e, y_e); // Contribution to material energy
   y_e *= rho;

   // Material energy nonlinear term
   rad_diff.H_form.Mult(x_e, z);
   y_e += z; // Contribution to material energy
   y_E.Set(-1, z); // Contribution to radiation energy

   // Radiation energy mass term
   rad_diff.L->Mult(x_E, z);
   y_E.Add(1 + c*dt*sigma, z); // Contribution to radiation energy
   y_e.Add(-c*dt*sigma, z); // Contribution to material energy

   // Radiation flux terms
   rad_diff.D->Mult(x_F, z);
   y_E += z;

   z.SetSize(n_rt);
   rad_diff.R->Mult(x_F, y_F);
   y_F *= -3*sigma/c/dt;

   rad_diff.Dt->Mult(x_E, z);
   y_F += z;
}

void BrunnerNowackIteration::Mult(const Vector &b, Vector &x) const
{
   const int maxit = 100;
   const double tol = 1e-6;

   std::cout << " Brunner-Nowack iteration\n"
             << " It.    Resnorm        Newton its.    Linear its.\n"
             << "=================================================\n";

   const int n_l2 = rad_diff.fes_l2.GetTrueVSize();
   const int n_rt = rad_diff.fes_rt.GetTrueVSize();
   const int n_eE = rad_diff.offsets[2];
   const int n_EF = rad_diff.offsets[3] - rad_diff.offsets[1];

   Vector x_eE(x, 0, n_eE);
   Vector x_EF(x, rad_diff.offsets[1], n_EF);

   Vector x_F(x, rad_diff.offsets[2], n_rt);

   r.SetSize(x.Size());
   Vector r_eE(r, 0, n_eE);
   Vector r_E(r, rad_diff.offsets[1], n_l2);
   Vector r_EF(r, rad_diff.offsets[1], n_EF);

   const Vector b_eE(const_cast<Vector&>(b), 0, n_eE);
   const Vector b_EF(const_cast<Vector&>(b), rad_diff.offsets[1], n_EF);

   c_eE.SetSize(n_eE);
   c_EF.SetSize(n_EF);

   for (int it = 0; it < maxit; ++it)
   {
      std::cout << " " << std::setw(3) << it << "    " << std::flush;
      // Compute full residual
      ApplyFullOperator(x, r);
      subtract(b, r, r); // Set r = b - J*x

      const double r_norm = r.Norml2();
      std::cout << std::setw(8) << std::scientific << r_norm << std::flush;
      if (r.Norml2() < tol)
      {
         std::cout << "       -" << std::endl;
         break;
      }

      // Modify right-hand side keeping radiation flux fixed
      r_eE = b_eE;
      z.SetSize(n_l2);
      rad_diff.D->Mult(x_F, z);
      r_E -= z;

      // Nonlinear solve for correction to x_e, x_E
      eE_solver.Mult(r_eE, x_eE);
      std::cout << "       " << eE_solver.GetNumIterations() << std::flush;

      // Compute residual again
      ApplyFullOperator(x, r);
      subtract(b, r, r); // Set r = b - J*x

      // Linear solve for correction to x_E, x_F
      c_EF = 0.0;
      EF_solver.Mult(r_EF, c_EF);
      std::cout << std::endl;

      // Update x given the correction c_EF
      x_EF += c_EF;
   }
}

void BrunnerNowackIteration::SetOperator(const Operator &op) { }

} // namespace mfem
