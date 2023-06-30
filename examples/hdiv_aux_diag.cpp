#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "hdiv_prolongation.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

struct AuxiliaryPreconditioner : Solver
{
   const Solver &S; // Smoother
   const Solver &B; // Auxiliary preconditioner
   const Operator &Pi; // Transfer operator
   const Array<int> &ess_dofs;
   mutable Vector z1, z2, w;

   AuxiliaryPreconditioner(const Solver &S_, const Solver &B_, const Operator &Pi_,
                           const Array<int> &ess_dofs_)
      : Solver(Pi_.Height()), S(S_), B(B_), Pi(Pi_), ess_dofs(ess_dofs_)
   { }

   void SetOperator(const Operator &op) { }

   void Mult(const Vector &b, Vector &x) const
   {
      z1.SetSize(Pi.Width());
      z2.SetSize(Pi.Width());

      Pi.MultTranspose(b, z1);
      B.Mult(z1, z2);
      Pi.Mult(z2, x);

      w.SetSize(b.Size());
      S.Mult(b, w);

      x += w;

      for (const int i : ess_dofs) { x[i] = b[i]; }
   }
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   // const char *mesh_file = "../data/star.mesh";
   const char *mesh_file = "../data/inline-quad.mesh";

   int order = 2;
   int ser_ref = 0;
   int par_ref = 0;
   int h_ref = 0;
   double kappa = 20.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.AddOption(&h_ref, "-hr", "--h-refinements",
                  "Number of multigrid refinements.");
   args.AddOption(&kappa, "-k", "--kappa", "IP-DG penalty parameter.");
   args.ParseCheck();

   kappa *= order*(order + 1);

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);

   const int dim = mesh.Dimension();
   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLobatto;
   RT_FECollection rt_fec(order-1, dim, b1, b2);
   FiniteElementSpace rt_fes(&mesh, &rt_fec);

   H1_FECollection h1_fec(order, dim, b1);
   FiniteElementSpace h1_fes(&mesh, &h1_fec, dim);

   cout << "RT DOFs: " << rt_fes.GetTrueVSize() << '\n';
   cout << "H1 DOFs: " << h1_fes.GetTrueVSize() << '\n';

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> h1_ess_dofs, rt_ess_dofs;
   h1_fes.GetEssentialTrueDofs(ess_bdr, h1_ess_dofs);
   rt_fes.GetEssentialTrueDofs(ess_bdr, rt_ess_dofs);

   DiscreteLinearOperator interp(&h1_fes, &rt_fes);
   interp.AddDomainInterpolator(new IdentityInterpolator);
   interp.Assemble();
   interp.Finalize();

   SparseMatrix Pi;
   interp.FormRectangularSystemMatrix(h1_ess_dofs, rt_ess_dofs, Pi);

   BilinearForm a_h1(&h1_fes);
   a_h1.AddDomainIntegrator(new VectorDiffusionIntegrator);
   a_h1.Assemble();
   a_h1.Finalize();
   a_h1.SetDiagonalPolicy(Operator::DIAG_ONE);
   SparseMatrix A_h1;
   a_h1.FormSystemMatrix(h1_ess_dofs, A_h1);

   BilinearForm a(&rt_fes);
   a.AddDomainIntegrator(new VectorFEDiffusionIntegrator);
   a.AddInteriorFaceIntegrator(new VectorFE_DGDiffusionIntegrator(kappa));
   a.AddBdrFaceIntegrator(new VectorFE_DGDiffusionIntegrator(kappa));
   a.Assemble();
   a.SetDiagonalPolicy(Operator::DIAG_ONE);
   SparseMatrix A;
   a.FormSystemMatrix(rt_ess_dofs, A);

   // auto save_matrix = [](const Operator &A, const string &fname)
   // {
   //    ofstream f(fname);
   //    A.PrintMatlab(f);
   // };

   // save_matrix(A_h1, "A_h1.txt");
   // save_matrix(A0, "A0.txt");
   // save_matrix(Pi, "Pi.txt");

   // AdditiveSchwarz as(S, C, I);

   // save_matrix(as, "as.txt");

   DSmoother A_diag_inv(A);
   UMFPackSolver A_h1_inv(A_h1);
   AuxiliaryPreconditioner aux(A_diag_inv, A_h1_inv, Pi, rt_ess_dofs);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(A);
   cg.SetPreconditioner(aux);

   Vector X(A.Height()), B(A.Height());

   B.Randomize(1);
   X = 0.0;
   cg.Mult(B, X);

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
