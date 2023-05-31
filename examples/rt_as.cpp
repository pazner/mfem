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
         // int idx = 0;
         // for (const int dof_s : dof_set)
         // {
         //    const int dof_i = (dof_s >= 0) ? dof_s : -1 - dof_s;
         //    dofs[idx] = dof_i;
         //    ++idx;
         // }
         cout << "DOF patch " << i << ": ";
         for (int j = 0; j < dofs.Size(); ++j)
         {
            cout << dofs[j];
            if (j < dofs.Size() - 1) { cout << ", "; }
            else { cout << '\n'; }
         }

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

struct PatchSmoother : Solver
{
   vec_ptr<Solver> patches_inv;
   vector<Array<int>> patch_dofs;
   const Array<int> &ess_dofs;
   mutable Vector z1, z2;

   PatchSmoother(const FiniteElementSpace &fes,
                 const SparseMatrix &A,
                 const Array<int> &ess_dofs_)
      : ess_dofs(ess_dofs_)
   {
      unique_ptr<Table> dof2el;
      {
         Table elem_dof_table = fes.GetElementToDofTable(); // deep copy
         {
            const int nnz = elem_dof_table.GetI()[elem_dof_table.Size()];
            int *J = elem_dof_table.GetJ();
            for (int i = 0; i < nnz; ++i)
            {
               const int j = J[i];
               J[i] = (j >= 0) ? j : -1 - j;
            }
         }
         dof2el.reset(Transpose(elem_dof_table));
      }

      set<int> ess_dof_set(ess_dofs.begin(), ess_dofs.end());

      unique_ptr<Table> v2el(fes.GetMesh()->GetVertexToElementTable());
      patch_dofs.resize(fes.GetNV());
      Array<int> el_dofs, row;
      for (int iv = 0; iv < fes.GetNV(); ++iv)
      {
         Array<int> &dofs = patch_dofs[iv];
         v2el->GetRow(iv, row);
         set<int> patch_el_set(row.begin(), row.end());
         for (const int iel : row)
         {
            fes.GetElementDofs(iel, el_dofs);
            for (const int s_dof : el_dofs)
            {
               const int i_dof = (s_dof >= 0) ? s_dof : -1 - s_dof;

               Array<int> dof_els;
               dof2el->GetRow(i_dof, dof_els);
               bool bdr_dof = false;
               for (const int iel2 : dof_els)
               {
                  if (patch_el_set.find(iel2) == patch_el_set.end())
                  {
                     // This dof belongs to a non-patch element. It is on the
                     // boundary of the patch.
                     bdr_dof = true;
                  }
               }
               if (ess_dof_set.find(i_dof) != ess_dof_set.end())
               {
                  bdr_dof = true;
               }

               if (!bdr_dof) { dofs.Append(i_dof); }
            }
         }
         dofs.Sort();
         dofs.Unique();

         cout << "Vertex " << iv << " patch size: " << dofs.Size() << '\n';

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

   {
      BilinearForm a_h1(&h1_fes);
      a_h1.AddDomainIntegrator(new VectorDiffusionIntegrator);
      a_h1.Assemble();
      a_h1.Finalize();
      SparseMatrix A_h1_2;
      a_h1.SetDiagonalPolicy(Operator::DIAG_ONE);
      a_h1.FormSystemMatrix(h1_ess_dofs, A_h1_2);
      {
         ofstream f("A_h1_2.txt");
         A_h1_2.PrintMatlab(f);
      }
      {
         ofstream f("A_h1.txt");
         A_h1->PrintMatlab(f);
      }
   }

   // PatchSmoother S(rt_fes, A, rt_ess_dofs);
   // DSmoother S(A);
   RT_ContinuityConstraints constraints(rt_fes);
   DofSmoother S(constraints, A, rt_ess_dofs);

   UMFPackSolver C(*A_h1);

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

   // HYPRE_BigInt row_starts[2] = {0, A.Height()};
   // HypreParMatrix A_hyp(MPI_COMM_WORLD, A.Height(), row_starts, &A);
   // HypreBoomerAMG amg(A_hyp);
   // X = 0.0;
   // cg.SetPreconditioner(amg);
   // cg.Mult(B, X);

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
