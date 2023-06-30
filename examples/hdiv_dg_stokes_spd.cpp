#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "hdiv_prolongation.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

struct AuxiliaryStokesSolver : Solver
{
   Operator &P_curl;
   Operator &M;
   Operator &A_h1_inv;
   Operator &K_inv;
   const Array<int> &ess_dofs;
   mutable Vector z1, z2, z3;

   AuxiliaryStokesSolver(
      Operator &P_curl_,
      Operator &M_,
      Operator &A_h1_inv_,
      Operator &K_inv_,
      const Array<int> &ess_dofs_)
      : Solver(P_curl_.Width()),
        P_curl(P_curl_),
        M(M_),
        A_h1_inv(A_h1_inv_),
        K_inv(K_inv_),
        ess_dofs(ess_dofs_)
   { }

   void SetOperator(const Operator &op) { }

   void Mult(const Vector &b, Vector &x) const
   {
      z1.SetSize(b.Size());
      z2.SetSize(P_curl.Height());
      z3.SetSize(P_curl.Height());

      A_h1_inv.Mult(b, z1);
      P_curl.Mult(z1, z2);
      M.Mult(z2, z3);
      K_inv.Mult(z3, z2);
      M.Mult(z2, z3);
      P_curl.MultTranspose(z3, z1);
      A_h1_inv.Mult(z1, x);

      for (const int i : ess_dofs) { x[i] = b[i]; }

      // x = b;
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

   H1_FECollection h1_fec(order, dim, b1);
   FiniteElementSpace h1_fes(&mesh, &h1_fec);
   // FiniteElementSpace h1_vec_fes(&mesh, &h1_fec, dim);

   RT_FECollection rt_fec(order-1, dim, b1, b2);
   FiniteElementSpace rt_fes(&mesh, &rt_fec);

   L2_FECollection l2_fec(order-1, dim, b1);
   FiniteElementSpace l2_fes(&mesh, &l2_fec);

   set<int> bdr_adj_elems;
   for (int ibe = 0; ibe < mesh.GetNBE(); ++ibe)
   {
      int el, info;
      mesh.GetBdrElementAdjacentElement(ibe, el, info);
      bdr_adj_elems.insert(el);
   }

   set<int> bdr_adj_dofs;
   for (int iel : bdr_adj_elems)
   {
      Array<int> dofs;
      h1_fes.GetElementDofs(iel, dofs);
      bdr_adj_dofs.insert(dofs.begin(), dofs.end());
   }

   {
      ofstream f("dofs.txt");
      for (int i : bdr_adj_dofs)
      {
         f << i << '\n';
      }
   }

   set<int> extended_elems;
   const Table &el2el = mesh.ElementToElementTable();
   for (int el : bdr_adj_elems)
   {
      Array<int> row;
      el2el.GetRow(el, row);
      extended_elems.insert(row.begin(), row.end());
   }

   for (int iel : extended_elems)
   {
      Array<int> dofs;
      h1_fes.GetElementDofs(iel, dofs);
      bdr_adj_dofs.insert(dofs.begin(), dofs.end());
   }

   {
      ofstream f("ext_dofs.txt");
      for (int i : bdr_adj_dofs)
      {
         f << i << '\n';
      }
   }

   {
      GridFunction v(&rt_fes);
      ifstream f("v.txt");
      v.Load(f, v.Size());

      ParaViewDataCollection pv("StokesAux", &mesh);
      pv.SetPrefixPath("ParaView");
      pv.SetHighOrderOutput(true);
      pv.SetLevelsOfDetail(order + 1);
      pv.RegisterField("v", &v);
      pv.SetCycle(0);
      pv.SetTime(0.0);
      pv.Save();
   }

   Array<int> ess_bdr;
   if (mesh.bdr_attributes.Size() > 0)
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
   }
   Array<int> rt_ess_dofs;
   Array<int> h1_ess_dofs;
   Array<int> l2_ess_dofs; // empty

   rt_fes.GetEssentialTrueDofs(ess_bdr, rt_ess_dofs);
   h1_fes.GetEssentialTrueDofs(ess_bdr, h1_ess_dofs);

   DiscreteLinearOperator curl(&h1_fes, &rt_fes);
   curl.AddDomainInterpolator(new CurlInterpolator);
   curl.Assemble();
   curl.Finalize();
   SparseMatrix P_curl;
   curl.FormRectangularSystemMatrix(h1_ess_dofs, rt_ess_dofs, P_curl);

   cout << "RT DOFs: " << rt_fes.GetTrueVSize() << '\n';

   BilinearForm k(&rt_fes);
   k.AddDomainIntegrator(new VectorFEDiffusionIntegrator);
   k.AddInteriorFaceIntegrator(new VectorFE_DGDiffusionIntegrator(kappa));
   k.AddBdrFaceIntegrator(new VectorFE_DGDiffusionIntegrator(kappa));
   k.SetDiagonalPolicy(Operator::DIAG_ONE);
   k.Assemble();
   SparseMatrix K;
   k.FormSystemMatrix(rt_ess_dofs, K);

   BilinearForm a_h1(&h1_fes);
   a_h1.AddDomainIntegrator(new DiffusionIntegrator);
   a_h1.Assemble();
   a_h1.Finalize();
   a_h1.SetDiagonalPolicy(Operator::DIAG_ONE);
   SparseMatrix A_h1;
   a_h1.FormSystemMatrix(h1_ess_dofs, A_h1);

   UMFPackSolver A_h1_inv(A_h1);
   UMFPackSolver K_inv(K);

   // RT_IP_AdditiveSchwarz as(rt_fes, ess_bdr, rt_ess_dofs, K);
   // UMFPackSolver as(K);

   MixedBilinearForm d(&rt_fes, &l2_fes);
   d.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   d.Assemble();
   SparseMatrix D, Dt;
   d.FormRectangularSystemMatrix(rt_ess_dofs, l2_ess_dofs, D);
   {
      unique_ptr<SparseMatrix> Dt_ptr(Transpose(D));
      Dt.Swap(*Dt_ptr);
   }

   Array<int> offsets({0, rt_fes.GetTrueVSize(), l2_fes.GetTrueVSize()});
   offsets.PartialSum();

   BlockOperator A(offsets);
   A.SetBlock(0, 0, &K);
   A.SetBlock(0, 1, &Dt, -1.0);
   A.SetBlock(1, 0, &D, -1.0);

   BilinearForm w(&l2_fes);
   w.AddDomainIntegrator(new MassIntegrator);
   w.Assemble();
   w.Finalize();
   SparseMatrix &W = w.SpMat();

   BilinearForm m(&rt_fes);
   m.AddDomainIntegrator(new VectorFEMassIntegrator);
   m.Assemble();
   m.Finalize();
   m.SetDiagonalPolicy(Operator::DIAG_ZERO);
   SparseMatrix M;
   m.FormSystemMatrix(rt_ess_dofs, M);
   // SparseMatrix &M = m.SpMat();

   auto save_matrix = [](const Operator &A, const string &fname)
   {
      ofstream f(fname);
      A.PrintMatlab(f);
   };

   save_matrix(P_curl, "P_curl.txt");
   save_matrix(K, "K.txt");
   save_matrix(W, "W.txt");
   save_matrix(M, "M.txt");
   save_matrix(D, "D.txt");
   save_matrix(A_h1, "A_h1.txt");

   auto save_array = [](const Array<int> &array, const string &fname)
   {
      ofstream f(fname);
      array.Print(f, 1);
   };

   save_array(rt_ess_dofs, "rt_ess_dofs.txt");
   save_array(h1_ess_dofs, "h1_ess_dofs.txt");

   DGMassInverse W_inv(l2_fes);

   BlockDiagonalPreconditioner P(offsets);
   P.SetDiagonalBlock(0, &K_inv);
   P.SetDiagonalBlock(1, &W_inv);

   BlockVector B(offsets);
   B.GetBlock(0).Randomize(1);
   B.GetBlock(0).SetSubVector(rt_ess_dofs, 0.0);
   B.GetBlock(1) = 0.0;

   BlockVector X(offsets);
   X = 0.0;

   MINRESSolver minres;
   minres.SetRelTol(1e-12);
   minres.SetMaxIter(2000);
   minres.SetPrintLevel(1);
   minres.SetOperator(A);
   minres.SetPreconditioner(P);
   minres.Mult(B, X);

   RAPOperator A_tilde(P_curl, K, P_curl);
   AuxiliaryStokesSolver aux(P_curl, M, A_h1_inv, K_inv, h1_ess_dofs);

   // save_matrix(aux, "aux.txt");

   Vector X_tilde(P_curl.Width()), B_tilde(P_curl.Width());
   P_curl.MultTranspose(B.GetBlock(0), B_tilde);
   X_tilde = 0.0;

   B_tilde.SetSubVector(h1_ess_dofs, 0.0);

   CGSolver cg;
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(200);
   cg.SetPrintLevel(1);
   cg.SetOperator(A_tilde);
   cg.SetPreconditioner(aux);
   cg.Mult(B_tilde, X_tilde);

   Vector X_1(P_curl.Height());
   P_curl.Mult(X_tilde, X_1);

   X_1 -= X.GetBlock(0);

   std::cout << "Difference (L infinity): " << X_1.Normlinf() << '\n';

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
