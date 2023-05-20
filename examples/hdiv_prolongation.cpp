#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "hdiv_prolongation.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

class RepeatedCoefficient : public VectorCoefficient
{
   Coefficient &coeff;
public:
   RepeatedCoefficient(int dim, Coefficient &coeff_)
      : VectorCoefficient(dim), coeff(coeff_)
   { }
   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      V.SetSize(vdim);
      V = coeff.Eval(T, ip);
   }
};

double u(const Vector &xvec);
double f(const Vector &xvec);

constexpr double pi = M_PI, pi2 = pi*pi;

double u(const Vector &xvec)
{
   int dim = xvec.Size();
   double x = pi*xvec[0], y = pi*xvec[1];
   if (dim == 2) { return sin(x)*sin(y); }
   else { double z = pi*xvec[2]; return sin(x)*sin(y)*sin(z); }
}

double f(const Vector &xvec)
{
   int dim = xvec.Size();
   double x = pi*xvec[0], y = pi*xvec[1];

   if (dim == 2)
   {
      return 2*pi2*sin(x)*sin(y);
   }
   else // dim == 3
   {
      double z = pi*xvec[2];
      return 3*pi2*sin(x)*sin(y)*sin(z);
   }
}

struct AdditivePreconditioner : Solver
{
   const HypreParMatrix &P;
   unique_ptr<HypreParMatrix> A0;
   // HypreBoomerAMG amg;
   SparseMatrix A0_diag;
   UMFPackSolver A0_inv;
   HypreSmoother D;

   mutable Vector z, b0, x0;

   AdditivePreconditioner(const HypreParMatrix &A,
                          const HypreParMatrix &P_)
      : P(P_),
        A0(RAP(&A, &P)),
        D(A, HypreSmoother::Jacobi),
        z(P.Height()),
        b0(P.Width()),
        x0(P.Width())
   {
      A0->GetDiag(A0_diag);
      A0_inv.SetOperator(A0_diag);
   }

   void SetOperator(const Operator &op) { }

   void Mult(const Vector &b, Vector &x) const
   {
      P.MultTranspose(b, b0);
      A0_inv.Mult(b0, x0);
      P.Mult(x0, x);

      D.Mult(b, z);

      x += z;
   }
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../data/star.mesh";
   const char *vis_vector = "";

   int order = 2;
   int ser_ref = 0;
   int par_ref = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   // For now, hard-code order = 2. Can generalize to higher-order once implemented.
   // args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.AddOption(&par_ref, "-vis", "--visualize", "Vector to visualize.");
   args.ParseCheck();

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);

   const int dim = mesh.Dimension();
   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLobatto;
   RT_FECollection fec(order-1, dim, b1, b2);
   ParFiniteElementSpace fes(&mesh, &fec);

   HYPRE_BigInt total_num_dofs = fes.GlobalTrueVSize();

   RT_ContinuityConstraints constraints(fes);

   cout << "Number of total DOFs:  " << total_num_dofs << '\n';
   cout << "Number of primary DOFs:" << constraints.n_primary_dofs << '\n';
   cout << "Number of entities:    " << constraints.entities.size() << '\n';

   SparseMatrix &P = constraints.GetProlongationMatrix();

   std::string vis_vec_str(vis_vector);
   if (!vis_vec_str.empty())
   {
      Vector X;
      ifstream f(vis_vec_str);
      X.Load(f);

      ParGridFunction x(&fes);
      P.Mult(X, x);

      ParaViewDataCollection pv("RTProlongation", &mesh);
      pv.SetPrefixPath("ParaView");
      pv.SetHighOrderOutput(true);
      pv.SetLevelsOfDetail(order + 1);
      pv.RegisterField("u", &x);
      pv.SetCycle(0);
      pv.SetTime(0.0);
      pv.Save();

      return 0;
   }

   {
      std::ofstream f("P.txt");
      P.PrintMatlab(f);
   }

   HYPRE_BigInt row_starts[2] = {0, P.Height()};
   HYPRE_BigInt col_starts[2] = {0, P.Width()};
   HypreParMatrix P_par(MPI_COMM_WORLD, P.Height(), P.Width(), row_starts,
                        col_starts, &P);

   Array<int> boundary_dofs;
   fes.GetBoundaryTrueDofs(boundary_dofs);

   FunctionCoefficient scalar_f_coeff(f), scalar_u_coeff(u);
   RepeatedCoefficient f_coeff(dim, scalar_f_coeff);
   RepeatedCoefficient u_coeff(dim, scalar_u_coeff);

   // 7. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   ParGridFunction x(&fes);
   x.ProjectCoefficient(u_coeff);

   // 8. Set up the linear form b(.) corresponding to the right-hand side.
   ParLinearForm b(&fes);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_coeff));
   b.Assemble();

   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new VectorFEDiffusionIntegrator);
   a.AddInteriorFaceIntegrator(new VectorFE_DGDiffusionIntegrator(20.0));
   a.AddBdrFaceIntegrator(new VectorFE_DGDiffusionIntegrator(20.0));
   a.Assemble();

   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying AMR constraints, parallel assembly, etc.
   HypreParMatrix A;
   a.FormSystemMatrix(boundary_dofs, A);
   A.Print("A.txt");

   Vector B(A.Height()), X(A.Height());
   B.Randomize(1);

   HypreBoomerAMG amg(A);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(amg);
   cg.SetOperator(A);

   X = 0.0;
   cg.Mult(B, X);

   X = 0.0;
   AdditivePreconditioner as(A, P_par);
   cg.SetPreconditioner(as);
   cg.Mult(B, X);


   // std::unique_ptr<HypreParMatrix> A0(RAP(&A, &P_par));
   // A0->Print("A0.txt");

   // HypreBoomerAMG amg(*A0);
   // CGSolver cg(MPI_COMM_WORLD);
   // cg.SetRelTol(1e-12);
   // cg.SetMaxIter(2000);
   // cg.SetPrintLevel(1);
   // cg.SetPreconditioner(amg);
   // cg.SetOperator(*A0);

   // Vector B(P.Width());
   // P.MultTranspose(b, B);

   // Vector X(P.Width());
   // X = 0.0;
   // cg.Mult(B, X);

   // P.Mult(X, x);

   ParaViewDataCollection pv("RTProlongation", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   pv.SetLevelsOfDetail(order + 1);
   pv.RegisterField("u", &x);
   pv.SetCycle(0);
   pv.SetTime(0.0);
   pv.Save();

   // X = 0.0;
   // for (int i = 0; i < X.Size(); ++ i)
   // {
   //    X[i] = 1.0;
   //    P.Mult(X, x);
   //    pv.SetCycle(pv.GetCycle() + 1);
   //    pv.SetTime(pv.GetTime() + 1);
   //    pv.Save();
   //    X[i] = 0.0;
   // }

   return 0;
}

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref)
{
   Mesh serial_mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   return mesh;
}
