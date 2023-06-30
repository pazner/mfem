#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "hdiv_prolongation.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

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
   const int b1 = BasisType::GaussLobatto;

   H1_FECollection h1_fec(order, dim, b1);
   FiniteElementSpace h1_fes(&mesh, &h1_fec, dim);

   // L2_FECollection l2_fec(order-2, dim, b1);
   L2_FECollection l2_fec(order-1, dim, b1);
   // H1_FECollection l2_fec(order-1, dim, b1);
   FiniteElementSpace l2_fes(&mesh, &l2_fec);

   cout << "H1 DOFs: " << h1_fes.GetTrueVSize() << '\n';
   cout << "L2 DOFs: " << l2_fes.GetTrueVSize() << '\n';

   Array<int> ess_bdr;
   if (mesh.bdr_attributes.Size() > 0)
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
   }
   // ess_bdr = 0;

   Array<int> h1_ess_dofs;
   Array<int> l2_ess_dofs; // empty
   h1_fes.GetEssentialTrueDofs(ess_bdr, h1_ess_dofs);

   BilinearForm k(&h1_fes);
   k.AddDomainIntegrator(new VectorDiffusionIntegrator);
   k.SetDiagonalPolicy(Operator::DIAG_ONE);
   k.Assemble();
   SparseMatrix K;
   k.FormSystemMatrix(h1_ess_dofs, K);

   MixedBilinearForm d(&h1_fes, &l2_fes);
   d.AddDomainIntegrator(new VectorDivergenceIntegrator);
   d.Assemble();
   SparseMatrix D, Dt;
   d.FormRectangularSystemMatrix(h1_ess_dofs, l2_ess_dofs, D);
   {
      unique_ptr<SparseMatrix> Dt_ptr(Transpose(D));
      Dt.Swap(*Dt_ptr);
   }

   MixedBilinearForm g{&l2_fes, &h1_fes};
   g.AddDomainIntegrator(new GradientIntegrator);
   g.Assemble();
   g.Finalize();
   SparseMatrix G;
   g.FormRectangularSystemMatrix(l2_ess_dofs, h1_ess_dofs, G);

   BilinearForm w(&l2_fes);
   w.AddDomainIntegrator(new MassIntegrator);
   w.Assemble();
   w.Finalize();
   SparseMatrix &W = w.SpMat();

   auto save_matrix = [](const Operator &A, const string &fname)
   {
      ofstream f(fname);
      A.PrintMatlab(f);
   };

   save_matrix(K, "K.txt");
   save_matrix(W, "W.txt");
   save_matrix(D, "D.txt");
   save_matrix(G, "G.txt");

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
