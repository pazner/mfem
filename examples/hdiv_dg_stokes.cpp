#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "hdiv_prolongation.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

template <typename T>
using vec_ptr = vector<unique_ptr<T>>;

struct DofSmoother : Solver
{
   vec_ptr<Solver> patches_inv;
   vector<Array<int>> patch_dofs;
   const Array<int> &ess_dofs;
   mutable Vector z1, z2;

   DofSmoother(const RT_ContinuityConstraints &constraints,
               const SparseMatrix &A,
               const Array<int> &ess_dofs_)
      : Solver(A.Width()), ess_dofs(ess_dofs_)
   {
      patch_dofs.resize(constraints.entities.size());

      for (int i = 0; i < constraints.entities.size(); ++i)
      {
         Array<int> &dofs = patch_dofs[i];
         const EntityDofs &e = constraints.entities[i];
         const set<int> &dof_set = e.GetGlobalDofSet();
         dofs.SetSize(dof_set.size());
         std::copy(dof_set.begin(), dof_set.end(), dofs.begin());

         // cout << "Patch " << i << " size " << dofs.Size() << '\n';
         // cout << "    ";
         // for (int j = 0; j < dofs.Size(); ++j)
         // {
         //    cout << dofs[j];
         //    if (j < dofs.Size() - 1) { cout << ", "; }
         //    else { cout << '\n'; }
         // }

         DenseMatrix submatrix(dofs.Size(), dofs.Size());
         A.GetSubMatrix(dofs, dofs, submatrix);

         patches_inv.emplace_back(new DenseMatrixInverse(submatrix));
      }
   }

   void SetOperator(const Operator &op) { }

   void Mult(const Vector &b, Vector &x) const
   {
      x = 0.0;
      for (int i = 0; i < patches_inv.size(); ++i)
      {
         const Array<int> &dofs = patch_dofs[i];
         if (dofs.Size() > 0)
         {
            z1.SetSize(dofs.Size());
            z2.SetSize(dofs.Size());

            b.GetSubVector(dofs, z1);
            patches_inv[i]->Mult(z1, z2);
            x.AddElementVector(dofs, z2);
         }
      }
   }

   void MultTranspose(const Vector &b, Vector &x) const { Mult(b, x); }
};

struct AdditiveSchwarz : Solver
{
   const Solver &S; // Smoother
   const Solver &C; // Coarse solver
   const Operator &P; // Prolongation
   mutable Vector z1, z2, z3;

   AdditiveSchwarz(const Solver &S_, const Solver &C_, const Operator &P_)
      : Solver(S_.Width()), S(S_), C(C_), P(P_)
   { }

   void SetOperator(const Operator &op) { }

   void Mult(const Vector &b, Vector &x) const
   {
      z1.SetSize(P.Width());
      z2.SetSize(P.Width());
      z3.SetSize(P.Height());

      P.MultTranspose(b, z1);
      C.Mult(z1, z2);
      P.Mult(z2, z3);

      S.Mult(b, x);
      x += z3;
   }
};

struct RT_IP_AdditiveSchwarz : Solver
{
   H1_FECollection h1_fec;
   FiniteElementSpace h1_fes;
   Array<int> h1_ess_dofs;
   DofSmoother S;
   DiscreteLinearOperator interp;
   SparseMatrix I;
   SparseMatrix A_h1;
   UMFPackSolver C;

   mutable Vector z1, z2, z3;

   RT_IP_AdditiveSchwarz(
      FiniteElementSpace &rt_fes,
      const Array<int> &ess_bdr,
      const Array<int> &rt_ess_dofs,
      const SparseMatrix &A
   ) : Solver(rt_fes.GetTrueVSize()),
      h1_fec(rt_fes.GetMaxElementOrder() - 1, rt_fes.GetMesh()->Dimension()),
      h1_fes(rt_fes.GetMesh(), &h1_fec, rt_fes.GetMesh()->Dimension()),
      S(RT_ContinuityConstraints(rt_fes), A, rt_ess_dofs),
      interp(&h1_fes, &rt_fes)
   {
      h1_fes.GetEssentialTrueDofs(ess_bdr, h1_ess_dofs);

      interp.AddDomainInterpolator(new IdentityInterpolator);
      interp.Assemble();
      interp.Finalize();
      interp.FormRectangularSystemMatrix(h1_ess_dofs, rt_ess_dofs, I);

      unique_ptr<SparseMatrix> A_h1_ptr(RAP(I, A, I));
      A_h1.Swap(*A_h1_ptr);

      A_h1.EliminateBC(h1_ess_dofs, SparseMatrix::DIAG_ONE);
      C.SetOperator(A_h1);
   }

   void SetOperator(const Operator &op) { }

   void Mult(const Vector &b, Vector &x) const
   {
      z1.SetSize(I.Width());
      z2.SetSize(I.Width());
      z3.SetSize(I.Height());

      I.MultTranspose(b, z1);
      C.Mult(z1, z2);
      I.Mult(z2, z3);

      S.Mult(b, x);
      x += z3;
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

   L2_FECollection l2_fec(order-1, dim, b1);
   FiniteElementSpace l2_fes(&mesh, &l2_fec);

   cout << "RT DOFs: " << rt_fes.GetTrueVSize() << '\n';

   Array<int> ess_bdr;

   if (mesh.bdr_attributes.Size() > 0)
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
   }
   // ess_bdr = 0;

   Array<int> rt_ess_dofs;
   Array<int> l2_ess_dofs; // empty
   rt_fes.GetEssentialTrueDofs(ess_bdr, rt_ess_dofs);

   BilinearForm k(&rt_fes);
   k.AddDomainIntegrator(new VectorFEDiffusionIntegrator);
   k.AddInteriorFaceIntegrator(new VectorFE_DGDiffusionIntegrator(kappa));
   k.AddBdrFaceIntegrator(new VectorFE_DGDiffusionIntegrator(kappa));
   k.SetDiagonalPolicy(Operator::DIAG_ONE);
   k.Assemble();
   SparseMatrix K;
   k.FormSystemMatrix(rt_ess_dofs, K);

   // RT_IP_AdditiveSchwarz as(rt_fes, ess_bdr, rt_ess_dofs, K);
   UMFPackSolver as(K);

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
   SparseMatrix M;
   m.FormSystemMatrix(rt_ess_dofs, M);

   auto save_matrix = [](const Operator &A, const string &fname)
   {
      ofstream f(fname);
      A.PrintMatlab(f);
   };

   save_matrix(K, "K.txt");
   save_matrix(M, "M.txt");
   save_matrix(W, "W.txt");
   save_matrix(D, "D.txt");

   // return 0;

   DGMassInverse M_inv(l2_fes);

   BlockDiagonalPreconditioner P(offsets);
   P.SetDiagonalBlock(0, &as);
   P.SetDiagonalBlock(1, &M_inv);

   MINRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(1e-12);
   krylov.SetMaxIter(2000);
   krylov.SetPrintLevel(1);
   krylov.SetOperator(A);
   krylov.SetPreconditioner(P);

   Vector X(A.Height()), B(A.Height());

   B.Randomize(1);
   B.SetSubVector(rt_ess_dofs, 0.0);
   for (int i = offsets[1]; i < offsets[2]; ++i) { B[i] = 0.0; }

   X = 0.0;
   krylov.Mult(B, X);

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
