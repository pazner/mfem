#include "mfem.hpp"
#include "fem/picojson.h"
#include "fem/integ/bilininteg_mass_kernels.hpp"
#include "fem/integ/bilininteg_diffusion_kernels.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class LocalEllipticProjection : public Operator
{
   SparseMatrix Pi;

   const Operator *P_hat;
   const Operator *R;

   mutable Vector w, z;

public:
   LocalEllipticProjection(const FiniteElementSpace &V, FiniteElementSpace &Vhat)
      : Operator(V.GetTrueVSize(), Vhat.GetTrueVSize()),
        Pi(V.GetVSize(), Vhat.GetVSize()),
        P_hat(Vhat.GetProlongationMatrix()),
        R(V.GetRestrictionOperator())
   {
      DiffusionIntegrator a;

      const int ne = V.GetNE();

      DenseMatrix a_1, a_2, a_e;
      DenseMatrix b_1, b_2;
      Array<int> dofs_1, dofs_2;

      for (int e = 0; e < ne; ++e)
      {
         const auto *fe_1 = V.GetFE(e);
         const auto *fe_2 = Vhat.GetFE(e);
         auto &T = *V.GetElementTransformation(e);

         a.AssembleElementMatrix(*fe_1, T, a_1);
         a.AssembleElementMatrix2(*fe_2, *fe_1, T, a_2);

         V.GetElementDofs(e, dofs_1);
         Vhat.GetElementDofs(e, dofs_2);

         const int order = fe_1->GetOrder();

         for (int i = 0; i < 3*order; ++i)
         {
            for (int j = 0; j < a_2.Width(); ++j)
            {
               a_2(i,j) = 0.0;
            }
            a_2(i,i) = 1.0;
         }

         a_e.SetSize(a_1.Height(), a_1.Width());
         a_e = 0.0;
         for (int i = 3*order; i < fe_1->GetDof(); ++i)
         {
            for (int j = 0; j < 3*order; ++j)
            {
               a_e(i,j) = -a_1(i,j);
               a_1(i,j) = 0.0;
               a_1(j,i) = 0.0;
            }
         }
         for (int i = 0; i < fe_1->GetDof(); ++i) { a_e(i,i) = 1.0; }
         for (int i = 0; i < 3*order; ++i)
         {
            a_1.SetRow(i, 0.0);
            a_1(i,i) = 1.0;
         }
         a_1.Invert();

         b_1.SetSize(a_e.Height(), a_2.Width());
         mfem::Mult(a_e, a_2, b_1);

         b_2.SetSize(a_1.Height(), b_1.Width());
         mfem::Mult(a_1, b_1, b_2);

         Pi.SetSubMatrix(dofs_1, dofs_2, b_2);
      }

      Pi.Finalize();

      w.SetSize(Vhat.GetVSize());
      z.SetSize(V.GetVSize());
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      if (P_hat) { P_hat->Mult(x, w); }
      else { w = x; }

      Pi.Mult(w, z);

      if (R) { R->Mult(z, y); }
      else { y = z; }
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      if (R) { R->MultTranspose(x, z); }
      else { z = x; }

      Pi.MultTranspose(z, w);

      if (P_hat) { P_hat->MultTranspose(w, y); }
      else { y = w; }
   }
};

struct FictitiousSolver : Solver
{
   const Solver &A_hat_inv;
   const Operator &R;
   const Array<int> ess_dofs;
   mutable Vector z1, z2;

   FictitiousSolver(const Solver &A_hat_inv_, const Operator &R_,
                    const Array<int> &ess_dofs_)
      : Solver(R_.Height()),
        A_hat_inv(A_hat_inv_),
        R(R_),
        ess_dofs(ess_dofs_)
   { }

   void SetOperator(const Operator &op) { }

   void Mult(const Vector &b, Vector &x) const
   {
      z1.SetSize(R.Width());
      z2.SetSize(R.Width());

      R.MultTranspose(b, z1);
      A_hat_inv.Mult(z1, z2);
      R.Mult(z2, x);

      for (int i : ess_dofs)
      {
         x[i] = b[i];
      }
   }
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 1. Parse command line options.
   string mesh_file = "../data/star.mesh";
   int order = 1;
   int ref = 0;
   string solver_type = "mumps";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ref, "-r", "--refine", "Number of refinements");
   args.AddOption(&solver_type, "-s", "--solver", "Solver. MUMPS or AMG.");
   args.ParseCheck();

   ParMesh mesh = [&]()
   {
      Mesh serial_mesh(mesh_file, 0, 0, true);
      for (int i = 0; i < ref; ++i) { serial_mesh.UniformRefinement(); }
      return ParMesh(MPI_COMM_WORLD, serial_mesh);
   }();

   const int dim = mesh.Dimension();

   H1_FECollection fec_h1(order, dim);
   ParFiniteElementSpace fes_h1(&mesh, &fec_h1);

   H1Duffy_FECollection fec_d(order, dim);
   ParFiniteElementSpace fes_d(&mesh, &fec_d);

   Array<int> ess_dofs_h1, ess_dofs_d;
   fes_h1.GetBoundaryTrueDofs(ess_dofs_h1);
   fes_d.GetBoundaryTrueDofs(ess_dofs_d);

   ParBilinearForm a_h1(&fes_h1);
   a_h1.SetDiagonalPolicy(Operator::DIAG_ONE);
   a_h1.AddDomainIntegrator(new DiffusionIntegrator);
   a_h1.AddDomainIntegrator(new MassIntegrator);
   a_h1.Assemble();

   const HYPRE_BigInt global_ndofs = fes_h1.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << global_ndofs << endl;
   }

   ParGridFunction x(&fes_h1);
   x = 0.0;

   FunctionCoefficient coeff([](const Vector &coo)
   {
      return exp(0.1*sin(5.1*coo[0] - 6.2*coo[1]) + 0.3*cos(4.3*coo[0] +3.4*coo[1]));
   });

   // ConstantCoefficient one(1.0);
   ParLinearForm b(&fes_h1);
   b.AddDomainIntegrator(new DomainLFIntegrator(coeff));
   b.Assemble();

   ParBilinearForm a_d(&fes_d);
   a_d.SetDiagonalPolicy(Operator::DIAG_ONE);
   a_d.AddDomainIntegrator(new DiffusionIntegrator);
   a_d.AddDomainIntegrator(new MassIntegrator);
   a_d.Assemble();

   HypreParMatrix A_d;
   a_d.FormSystemMatrix(ess_dofs_d, A_d);

   HypreParMatrix A_h1;
   Vector B, X;
   a_h1.FormLinearSystem(ess_dofs_h1, x, b, A_h1, X, B);

   unique_ptr<Solver> prec;
   if (solver_type == "amg")
   {
      auto amg = make_unique<HypreBoomerAMG>(A_d);

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
      MFEM_ABORT("");
      // prec.reset(new MUMPSSolver(A_lor));
   }
   else
   {
      MFEM_ABORT("Unknown solver type");
   }

   LocalEllipticProjection R(fes_h1, fes_d);
   FictitiousSolver aux(*prec, R, ess_dofs_h1);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(A_h1);
   cg.SetPreconditioner(aux);

   tic_toc.Restart();
   cg.Mult(B, X);
   const double elapsed = tic_toc.RealTime();

   // Vector Y(B.Size());
   // A_h1.Mult(X, Y);
   // Y -= B;
   // cout << Y.Normlinf() << '\n';

   const int niter = cg.GetNumIterations();

   picojson::object results;
   results["niter"] = picojson::value(double(niter));
   results["ndofs"] = picojson::value(double(global_ndofs));
   results["solver"] = picojson::value(solver_type);
   results["N"] = picojson::value(double(order));
   results["mesh"] = picojson::value(mesh_file);
   results["ref"] = picojson::value(double(ref));
   results["nnz_h1"] = picojson::value(double(A_h1.NNZ()));
   results["nnz_d"] = picojson::value(double(A_d.NNZ()));
   results["elapsed"] = picojson::value(elapsed);

   if (Mpi::Root())
   {
      cout << "=== " << picojson::value(results) << '\n';
   }

   return 0;
}
