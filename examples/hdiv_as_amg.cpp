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
      : ess_dofs(ess_dofs_)
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
      for (const int i : ess_dofs)
      {
         x[i] = b[i];
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
      : S(S_), C(C_), P(P_)
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

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);

   const int dim = mesh.Dimension();
   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLobatto;
   RT_FECollection rt_fec(order-1, dim, b1, b2);
   FiniteElementSpace rt_fes(&mesh, &rt_fec);

   H1_FECollection h1_fec(order - 1, dim, b1);
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

   SparseMatrix I;
   interp.FormRectangularSystemMatrix(h1_ess_dofs, rt_ess_dofs, I);

   BilinearForm a(&rt_fes);
   a.AddDomainIntegrator(new VectorFEDiffusionIntegrator);
   a.AddInteriorFaceIntegrator(new VectorFE_DGDiffusionIntegrator(kappa));
   a.AddBdrFaceIntegrator(new VectorFE_DGDiffusionIntegrator(kappa));
   a.Assemble();
   SparseMatrix A;
   a.FormSystemMatrix(rt_ess_dofs, A);

   unique_ptr<SparseMatrix> A_h1(RAP(I, A, I));

   // BCs...
   A.EliminateBC(rt_ess_dofs, SparseMatrix::DIAG_ONE);
   A_h1->EliminateBC(h1_ess_dofs, SparseMatrix::DIAG_ONE);

   RT_ContinuityConstraints constraints(rt_fes);
   DofSmoother S(constraints, A, rt_ess_dofs);

   HYPRE_BigInt row_starts[2] = {0, A_h1->Height()};
   HypreParMatrix A_h1_hyp(MPI_COMM_WORLD, A_h1->Height(), row_starts, A_h1.get());
   HypreBoomerAMG C(A_h1_hyp);
   // UMFPackSolver C(*A_h1);

   AdditiveSchwarz as(S, C, I);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(A);
   cg.SetPreconditioner(as);

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
