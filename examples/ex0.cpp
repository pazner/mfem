//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Poisson
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../data/star.mesh";
   int order = 1;
   int ref = 0;
   bool duffy = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ref, "-r", "--refine", "Number of refinements");
   args.AddOption(&duffy, "-d", "--duffy", "-no-d", "--no-duffy", "Use Duffy?");
   args.ParseCheck();

   Mesh mesh(mesh_file, 0, 0, true);
   // Mesh mesh_orig(mesh_file, 0, 0, false);
   // Mesh mesh = Mesh::MakeSimplicial(mesh_orig);

   for (int i = 0; i < ref; ++i) { mesh.UniformRefinement(); }

   // mesh.UniformRefinement();
   const int dim = mesh.Dimension();

   Mesh mesh_lor = [&]()
   {
      if (duffy)
      {
         return Mesh::MakeDuffyRefined(mesh, order, BasisType::GaussLobatto);
      }
      else
      {
         return Mesh::MakeRefined(mesh, order, BasisType::GaussLobatto);
      }
   }();
   mesh_lor.Save("mesh_lor.mesh");

   H1_FECollection fec_lor(1, dim);
   FiniteElementSpace fes_lor(&mesh_lor, &fec_lor);

   BilinearForm a_lor(&fes_lor);
   a_lor.SetDiagonalPolicy(Operator::DIAG_ONE);
   a_lor.AddDomainIntegrator(new DiffusionIntegrator);
   a_lor.AddDomainIntegrator(new MassIntegrator);
   a_lor.Assemble();

   Array<int> ess_dofs_lor;
   fes_lor.GetBoundaryTrueDofs(ess_dofs_lor);

   SparseMatrix A_lor;
   a_lor.FormSystemMatrix(ess_dofs_lor, A_lor);

   // {
   //    ofstream f("A_lor.txt");
   //    A_lor.PrintMatlab(f);
   // }

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   unique_ptr<FiniteElementCollection> fec;
   if (duffy)
   {
      fec.reset(new H1Duffy_FECollection(order, dim));
   }
   else
   {
      fec.reset(new H1_FECollection(order, dim));
   }

   FiniteElementSpace fespace(&mesh, fec.get());
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // 4. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   // 5. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   FunctionCoefficient coeff([](const Vector &coo)
   {
      return exp(0.1*sin(5.1*coo[0] - 6.2*coo[1]) + 0.3*cos(4.3*coo[0] +3.4*coo[1]));
   });

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   LinearForm b(&fespace);
   // b.AddDomainIntegrator(new DomainLFIntegrator(coeff, &ir));
   // b.AddDomainIntegrator(new DomainLFIntegrator(one, &ir));
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&fespace);
   a.SetDiagonalPolicy(Operator::DIAG_ONE);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.AddDomainIntegrator(new MassIntegrator);
   // a.AddDomainIntegrator(new DiffusionIntegrator(&ir));
   // a.AddDomainIntegrator(new MassIntegrator(&ir));
   // a.AddDomainIntegrator(new MassIntegrator(&ir));
   a.Assemble();

   // 8. Form the linear system A X = B. This includes eliminating boundary
   //    conditions, applying AMR constraints, and other transformations.
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   // GSSmoother M(A);
   UMFPackSolver M(A_lor);
   PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

   // {
   //    // string fname = duffy ? "M_d.txt" : "M_h.txt";
   //    string fname = "A.txt";
   //    ofstream f(fname);
   //    A.PrintMatlab(f);
   // }

   // 10. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
   a.RecoverFEMSolution(X, b, x);

   ParaViewDataCollection pv("DuffyEx0", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   // pv.SetLevelsOfDetail(2*order);
   pv.SetLevelsOfDetail(std::min(8, 2*order));
   pv.RegisterField("u", &x);
   pv.Save();

   return 0;
}
