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

#include "general/forall.hpp"

namespace mfem
{

LinearizedEnergyOperator::LinearizedEnergyOperator(
   RadiationDiffusionOperator &rad_diff_)
   : Operator(rad_diff_.offsets[2]),
     rad_diff(rad_diff_),
     dH(nullptr) { }

void LinearizedEnergyOperator::Mult(const Vector &x, Vector &y) const
{
   using namespace MMS;

   const int n_l2 = rad_diff.fes_l2.GetTrueVSize();
   const Array<int> &offsets = rad_diff.GetOffsets();

   const Vector x_e(const_cast<Vector&>(x), offsets[0], n_l2);
   const Vector x_E(const_cast<Vector&>(x), offsets[1], n_l2);

   Vector y_e(y, offsets[0], n_l2);
   Vector y_E(y, offsets[1], n_l2);

   z.SetSize(n_l2);

   // Material energy mass matrix
   rad_diff.L->Mult(x_e, z);
   y_e.Set(rho, z);

   // Material energy linearized term
   dH->Mult(x_e, z);
   y_e.Add(1.0, z);
   y_E.Set(-1.0, z);

   // Radiation energy mass matrix
   rad_diff.L->Mult(x_E, z);
   y_E.Add(1 + c*dt*sigma, z);
   y_e.Add(-c*dt*sigma, z);

   y_e.SyncAliasMemory(y);
   y_E.SyncAliasMemory(y);
}

NonlinearEnergyOperator::NonlinearEnergyOperator(
   RadiationDiffusionOperator &rad_diff_)
   : Operator(rad_diff_.offsets[2]),
     rad_diff(rad_diff_),
     linearized_op(rad_diff),
     dt(0.0)
{ }

void NonlinearEnergyOperator::Mult(const Vector &x, Vector &y) const
{
   using namespace MMS;

   const int n_l2 = rad_diff.fes_l2.GetTrueVSize();
   const Array<int> &offsets = rad_diff.GetOffsets();

   const Vector x_e(const_cast<Vector&>(x), offsets[0], n_l2);
   const Vector x_E(const_cast<Vector&>(x), offsets[1], n_l2);

   Vector y_e(y, offsets[0], n_l2);
   Vector y_E(y, offsets[1], n_l2);

   // Material energy mass term
   z.SetSize(n_l2);
   rad_diff.L->Mult(x_e, y_e); // Contribution to material energy
   y_e *= rho;

   // Material energy nonlinear term
   rad_diff.H.Mult(x_e, z);
   y_e += z; // Contribution to material energy
   y_E.Set(-1, z); // Contribution to radiation energy

   // Radiation energy mass term
   rad_diff.L->Mult(x_E, z);
   y_E.Add(1 + c*dt*sigma, z); // Contribution to radiation energy
   y_e.Add(-c*dt*sigma, z); // Contribution to material energy

   y_e.SyncAliasMemory(y);
   y_E.SyncAliasMemory(y);
}

Operator &NonlinearEnergyOperator::GetGradient(const Vector &x) const
{
   const int n_l2 = rad_diff.fes_l2.GetTrueVSize();
   const Array<int> &offsets = rad_diff.GetOffsets();

   // Linearize the nonlinear H operator about x_e
   const Vector x_e(const_cast<Vector&>(x), offsets[0], n_l2);
   const Operator &dH = rad_diff.H.GetGradient(x_e);
   linearized_op.SetLinearizedMaterialOperator(dH);
   linearized_op.SetTimeStep(dt);

   // Return the block linearized operator
   const Operator &op = linearized_op;
   return const_cast<Operator&>(op);
}

void NonlinearEnergyOperator::Setup(const double dt_)
{
   dt = dt_;
   rad_diff.H.SetTimeStep(dt);
   rad_diff.H.SetMaterialEnergy(rad_diff.e_gf);
}

void EnergyBlockJacobi::Mult(const Vector &x, Vector &y) const
{
   const int n_l2 = diag_dH.Size();

   const double *x_e = x.Read();
   const double *x_E = x_e + n_l2;

   double *y_e = y.Write();
   double *y_E = y_e + n_l2;

   const double *J = block_jacobi.Read();

   MFEM_FORALL(i, n_l2,
   {
      double x_e_i = x_e[i];
      double x_E_i = x_E[i];

      y_e[i] = J[0 + 4*i]*x_e_i + J[2 + 4*i]*x_E_i;
      y_E[i] = J[1 + 4*i]*x_e_i + J[3 + 4*i]*x_E_i;
   });
}

void EnergyBlockJacobi::SetOperator(const Operator &op)
{
   auto *linearized_op = dynamic_cast<const LinearizedEnergyOperator*>(&op);
   MFEM_VERIFY(linearized_op != nullptr, "Wrong operator type.");
   const RadiationDiffusionOperator &rad_diff = linearized_op->rad_diff;

   const int n_l2 = rad_diff.fes_l2.GetTrueVSize();

   diag_dH.SetSize(n_l2);
   linearized_op->dH->AssembleDiagonal(diag_dH);

   const double dt = linearized_op->dt;

   block_jacobi.SetSize(4*n_l2);

   using namespace MMS;

   const double *d_L = rad_diff.diag_L.Read();
   const double *d_dH = diag_dH.Read();
   double *d_J = block_jacobi.Write();

   MFEM_FORALL(i, n_l2,
   {
      const double L = d_L[i];
      const double dH = d_dH[i];

      const double J_11 = rho*L + dH;
      const double J_12 = -c*dt*sigma*L;
      const double J_21 = -dH;
      const double J_22 = (1 + c*dt*sigma)*L;

      const double det_J_inv = 1.0/(J_11*J_22 - J_12*J_21);

      d_J[0 + 4*i] = J_22*det_J_inv; // (1,1)
      d_J[1 + 4*i] = -J_21*det_J_inv; // (2,1)
      d_J[2 + 4*i] = -J_12*det_J_inv; // (1,2)
      d_J[3 + 4*i] = J_11*det_J_inv; // (2,2)
   });
}

RadiationDiffusionLinearSolver::RadiationDiffusionLinearSolver(
   RadiationDiffusionOperator &rad_diff)
   : solver(rad_diff.GetMesh(),
            rad_diff.GetRTSpace(),
            rad_diff.GetL2Space(),
            L_coeff,
            R_coeff,
            rad_diff.empty,
            L2CoefficientMode::RECIPROCAL),
     dt_prev(0.0)
{ }

void RadiationDiffusionLinearSolver::Setup(double dt)
{
   using namespace MMS;
   // On the first call, dt_prev == 0.0, so the setup always runs the first time.
   if (dt != dt_prev)
   {
      // Set up/assemble the linear solver if the time step changes.
      L_coeff.constant = 1.0 + c*dt*sigma;
      R_coeff.constant = 3.0*sigma/c/dt;
      dt_prev = dt;
      solver.Setup();
   }
}

BrunnerNowackIteration::BrunnerNowackIteration(
   RadiationDiffusionOperator &rad_diff_)
   : IterativeSolver(rad_diff_.GetComm()),
     rad_diff(rad_diff_),
     N_eE(rad_diff),
     J_eE_solver(rad_diff.GetComm()),
     eE_solver(rad_diff.GetComm()),
     EF_solver(rad_diff)
{
   height = width = rad_diff.Height();

   J_eE_solver.SetMaxIter(1000);
   J_eE_solver.SetRelTol(1e-14);
   J_eE_solver.SetAbsTol(1e-14);
   J_eE_solver.SetPrintLevel(IterativeSolver::PrintLevel().None());
   J_eE_solver.SetPreconditioner(J_eE_preconditioner);

   eE_solver.iterative_mode = false;
   eE_solver.SetMaxIter(20);
   eE_solver.SetRelTol(1e-8);
   eE_solver.SetAbsTol(1e-8);
   eE_solver.SetOperator(N_eE);
   eE_solver.SetSolver(J_eE_solver);
   eE_solver.SetPrintLevel(IterativeSolver::PrintLevel().None());

   int rank;
   MPI_Comm_rank(rad_diff.GetComm(), &rank);
   if (rank == 0) { print_options.Iterations(); }
   else { print_options.None(); }
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
   rad_diff.H.Mult(x_e, z);

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

   y_e.SyncAliasMemory(y);
   y_E.SyncAliasMemory(y);
   y_F.SyncAliasMemory(y);
}

void BrunnerNowackIteration::Mult(const Vector &b, Vector &x) const
{
   const int maxit = 100;
   const double tol = 1e-6;

   const bool print = print_options.iterations;

   if (print)
   {
      std::cout << " It.    Resnorm        Newton its.    Linear its.\n"
                << "=================================================\n"
                << std::scientific << std::left << std::setprecision(2);
   }

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

   auto sync = [](Vector &v1, const Vector &v2) { v1.SyncAliasMemory(v2); };
   auto sync_x = [&]() { sync(x_eE, x); sync(x_EF, x); sync(x_F, x); };
   auto sync_r = [&]() { sync(r_eE, r); sync(r_E, r); sync(r_EF, r); };
   auto sync_xr = [&]() { sync_x(); sync_r(); };

   const double b_norm = Norm(b);

   for (int it = 0; it < maxit; ++it)
   {
      if (print)
      {
         std::cout << " " << std::setw(3) << it << "    " << std::flush;
      }
      // Compute full residual
      sync_xr();
      ApplyFullOperator(x, r);
      subtract(b, r, r); // Set r = b - J*x

      const double r_norm = Norm(r);
      if (print) { std::cout << std::setw(15) << r_norm << std::flush; }
      if (r_norm/b_norm < tol)
      {
         if (print) { std::cout << "-\n"; }
         break;
      }

      // Modify right-hand side keeping radiation flux fixed
      r_eE = b_eE;
      z.SetSize(n_l2);
      rad_diff.D->Mult(x_F, z);

      // r_eE has been modified, update r_E
      r_eE.SyncAliasMemory(r);
      r_E.SyncMemory(r);

      r_E -= z;

      // r_E has been modified, update r_eE
      r_E.SyncAliasMemory(r);
      r_eE.SyncMemory(r);

      // Nonlinear solve for correction to x_e, x_E
      eE_solver.Mult(r_eE, x_eE);
      if (print )
      {
         std::cout << std::setw(15) << eE_solver.GetNumIterations() << std::flush;
      }

      // Compute residual again
      sync_xr();
      ApplyFullOperator(x, r);
      subtract(b, r, r); // Set r = b - J*x

      // Linear solve for correction to x_E, x_F
      r_EF.SyncMemory(r);
      EF_solver.Mult(r_EF, c_EF);
      if (print) { std::cout << EF_solver.GetNumIterations() << std::endl; }

      // Update x given the correction c_EF
      x_EF += c_EF;
   }
   if (print) { std::cout << std::endl; }

   sync_x();
}

void BrunnerNowackIteration::Setup(double dt)
{
   N_eE.Setup(dt);
   EF_solver.Setup(dt);
}

void BrunnerNowackIteration::SetOperator(const Operator &op) { }

} // namespace mfem
