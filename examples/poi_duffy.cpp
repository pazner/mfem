#include "mfem.hpp"
#include "fem/picojson.h"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 1. Parse command line options.
   string mesh_file = "../data/star.mesh";
   int order = 1;
   int ref = 0;
   bool duffy = false;
   bool lor = false;
   bool vis = false;
   string solver_type = "mumps";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ref, "-r", "--refine", "Number of refinements");
   args.AddOption(&solver_type, "-s", "--solver", "Solver. MUMPS or AMG.");
   args.AddOption(&duffy, "-d", "--duffy", "-no-d", "--no-duffy", "Use Duffy?");
   args.AddOption(&lor, "-l", "--lor", "-no-l", "--no-lor", "Solver LOR system?");
   args.AddOption(&vis, "-v", "--vis", "-no-v", "--no-vis", "ParaView vis?");
   args.ParseCheck();

   ParMesh mesh = [&]()
   {
      Mesh serial_mesh(mesh_file, 0, 0, true);
      for (int i = 0; i < ref; ++i) { serial_mesh.UniformRefinement(); }
      return ParMesh(MPI_COMM_WORLD, serial_mesh);
   }();

   const int dim = mesh.Dimension();

   ParMesh mesh_lor = [&]()
   {
      if (duffy)
      {
         return ParMesh::MakeDuffyRefined(mesh, order, BasisType::GaussLobatto);
      }
      else
      {
         return ParMesh::MakeRefined(mesh, order, BasisType::GaussLobatto);
      }
   }();

   H1_FECollection fec_lor(1, dim);
   ParFiniteElementSpace fes_lor(&mesh_lor, &fec_lor);

   ParBilinearForm a_lor(&fes_lor);
   a_lor.SetDiagonalPolicy(Operator::DIAG_ONE);
   a_lor.AddDomainIntegrator(new DiffusionIntegrator);
   a_lor.AddDomainIntegrator(new MassIntegrator);
   a_lor.Assemble();

   Array<int> ess_dofs_lor;
   fes_lor.GetBoundaryTrueDofs(ess_dofs_lor);

   HypreParMatrix A_lor;
   a_lor.FormSystemMatrix(ess_dofs_lor, A_lor);

   unique_ptr<FiniteElementCollection> fec;
   if (duffy)
   {
      fec.reset(new H1Duffy_FECollection(order, dim));
   }
   else
   {
      fec.reset(new H1_FECollection(order, dim));
   }

   ParFiniteElementSpace fespace(&mesh, fec.get());
   const HYPRE_BigInt global_ndofs = fespace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << global_ndofs << endl;
   }

   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   ParGridFunction x(&fespace);
   x = 0.0;

   FunctionCoefficient coeff([](const Vector &coo)
   {
      return exp(0.1*sin(5.1*coo[0] - 6.2*coo[1]) + 0.3*cos(4.3*coo[0] +3.4*coo[1]));
   });

   ConstantCoefficient one(1.0);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   ParBilinearForm a(&fespace);
   a.SetDiagonalPolicy(Operator::DIAG_ONE);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.AddDomainIntegrator(new MassIntegrator);
   a.Assemble();

   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   unique_ptr<Solver> prec;
   if (solver_type == "amg" || solver_type == "ho-amg")
   {
      unique_ptr<HypreBoomerAMG> amg;
      if (solver_type == "amg")
      {
         amg = make_unique<HypreBoomerAMG>(A_lor);
      }
      else
      {
         amg = make_unique<HypreBoomerAMG>(A);
      }

      HYPRE_BoomerAMGSetCoarsenType(*amg, 6);
      HYPRE_BoomerAMGSetInterpType(*amg, 0);

      HYPRE_BoomerAMGSetCycleRelaxType(*amg, 89, 1);
      HYPRE_BoomerAMGSetCycleRelaxType(*amg, 89, 2);
      HYPRE_BoomerAMGSetCycleRelaxType(*amg, 9, 3);

      HYPRE_BoomerAMGSetNumSweeps(*amg, 2);

      HYPRE_BoomerAMGSetAggNumLevels(*amg, 0);

      HYPRE_BoomerAMGSetStrongThreshold(*amg, 0.5);

      prec = std::move(amg);
   }
   else if (solver_type == "mumps")
   {
      prec.reset(new MUMPSSolver(A_lor));
   }
   else
   {
      MFEM_ABORT("Unknown solver type");
   }

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   if (lor)
   {
      cg.SetOperator(A_lor);
   }
   else
   {
      cg.SetOperator(A);
   }
   cg.SetPreconditioner(*prec);

   tic_toc.Restart();
   cg.Mult(B, X);
   const double elapsed = tic_toc.RealTime();

   const int niter = cg.GetNumIterations();

   picojson::object results;
   results["niter"] = picojson::value(double(niter));
   results["ndofs"] = picojson::value(double(global_ndofs));
   results["solver"] = picojson::value(solver_type);
   results["N"] = picojson::value(double(order));
   results["mesh"] = picojson::value(mesh_file);
   results["ref"] = picojson::value(double(ref));
   results["nnz"] = picojson::value(double(A.NNZ()));
   results["nnz_lor"] = picojson::value(double(A_lor.NNZ()));
   results["elapsed"] = picojson::value(elapsed);

   if (Mpi::Root())
   {
      cout << "=== " << picojson::value(results) << '\n';
   }

   if (vis)
   {
      a.RecoverFEMSolution(X, b, x);
      ParaViewDataCollection pv("PoiDuffy", &mesh);
      pv.SetPrefixPath("ParaView");
      pv.SetHighOrderOutput(true);
      pv.SetLevelsOfDetail(std::min(8, 2*order));
      pv.RegisterField("u", &x);
      pv.Save();
   }

   return 0;
}
