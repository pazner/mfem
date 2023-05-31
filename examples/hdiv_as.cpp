#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "hdiv_prolongation.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

template <typename T>
using vec_ptr = vector<unique_ptr<T>>;

struct PatchSmoother : Solver
{
   vec_ptr<Solver> patches_inv;
   vector<Array<int>> patch_dofs;
   const Array<int> &ess_dofs;
   mutable Vector z1, z2;

   PatchSmoother(const RT_ContinuityConstraints &constraints,
                 const SparseMatrix &A)
      : ess_dofs(constraints.bdr_dofs)
   {
      const FiniteElementSpace &fes = constraints.fes;
      const SparseMatrix &P = constraints.GetProlongationMatrix();
      set<int> ess_dof_set(ess_dofs.begin(), ess_dofs.end());

      unique_ptr<Table> v2el(fes.GetMesh()->GetVertexToElementTable());
      patch_dofs.resize(fes.GetNV());
      Array<int> el_dofs, row;
      for (int iv = 0; iv < fes.GetNV(); ++iv)
      {
         Array<int> &dofs = patch_dofs[iv];
         v2el->GetRow(iv, row);

         std::set<int> patch_el_set(row.begin(), row.end());

         for (const int iel : row)
         {
            fes.GetElementDofs(iel, el_dofs);
            for (const int s_dof : el_dofs)
            {
               const int i_dof = (s_dof >= 0) ? s_dof : -1 - s_dof;

               Array<int> col_idx;
               Vector vals;
               P.GetRow(i_dof, col_idx, vals);
               for (int i = 0; i < col_idx.Size(); ++i)
               {
                  // Is this an interior boundary dof? That means that the DOF
                  // is shared with an element outside of the patch.
                  bool bdr_dof = false;
                  const int entity_idx = constraints.dof2entity[col_idx[i]];

                  for (const ConstrainedElementEntity &c :
                       constraints.entities[entity_idx].constrained)
                  {
                     const int iel2 = c.dofs.element_index;
                     if (patch_el_set.find(iel2) == patch_el_set.end())
                     {
                        bdr_dof = true;
                        break;
                     }
                  }
                  if (ess_dof_set.find(col_idx[i]) != ess_dof_set.end())
                  {
                     bdr_dof = true;
                  }
                  if (bdr_dof) { continue; }

                  if (std::abs(vals[i]) > 1e-10)
                  {
                     dofs.Append(col_idx[i]);
                  }
                  // dofs.Append(col_idx);
               }
            }
         }
         dofs.Sort();
         dofs.Unique();

         std::cout << "Vertex " << iv << " patch size: " << dofs.Size() << '\n';

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
      // x *= 1.0/5.0;
      // x *= 1.0/10.0;

      for (const int i : ess_dofs)
      {
         x[i] = b[i];
      }
   }

   void MultTranspose(const Vector &b, Vector &x) const { Mult(b, x); }
};

struct H1_Intersect_RT
{
   BilinearForm a;
   RT_ContinuityConstraints constraints;
   SparseMatrix A;
   SparseMatrix *P;
   unique_ptr<SparseMatrix> A0;
   SparseMatrix A0_diag;
   unique_ptr<Solver> S;
   Array<int> bdr_dofs;

   HYPRE_BigInt row_starts[2];
   HYPRE_BigInt col_starts[2];

   H1_Intersect_RT(ParFiniteElementSpace &fes, const Array<int> &ess_bdr,
                   bool coarse)
      : a(&fes),
        constraints(fes)
   {
      a.AddDomainIntegrator(new VectorFEDiffusionIntegrator);
      // a.AddInteriorFaceIntegrator(new VectorFE_DGDiffusionIntegrator(20.0));
      // a.AddBdrFaceIntegrator(new VectorFE_DGDiffusionIntegrator(20.0));
      a.AddDomainIntegrator(new VectorFEMassIntegrator);
      a.Assemble();

      // fes.GetEssentialTrueDofs(ess_bdr, bdr_dofs);
      a.FormSystemMatrix(bdr_dofs, A);

      P = &constraints.GetProlongationMatrix();
      A0.reset(RAP(*P, A, *P));

      A0->EliminateBC(constraints.bdr_dofs, SparseMatrix::DIAG_ONE);

      if (coarse)
      {
         // S.reset(new HypreBoomerAMG(*A0));

         S.reset(new UMFPackSolver(*A0));
      }
      else
      {
         // HypreSmoother *smoother = new HypreSmoother(*A0, HypreSmoother::l1GS);
         // HypreSmoother *smoother = new GSSmoother()
         // smoother->SetOperatorSymmetry(true);
         // auto *smoother = new SymmetricSmoother<GSSmoother>(*A0);
         PatchSmoother *smoother = new PatchSmoother(constraints, *A0);
         // UMFPackSolver *smoother = new UMFPackSolver(*A0);
         S.reset(smoother);
      }
   }
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

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   // For now, hard-code order = 2. Can generalize to higher-order once implemented.
   // args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.AddOption(&h_ref, "-hr", "--h-refinements",
                  "Number of multigrid refinements.");
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
   Array<int> h1_ess_dofs;
   h1_fes.GetEssentialTrueDofs(ess_bdr, h1_ess_dofs);

   RT_ContinuityConstraints constraints(rt_fes);

   DiscreteLinearOperator interp(&h1_fes, &rt_fes);
   interp.AddDomainInterpolator(new IdentityInterpolator);
   interp.Assemble();
   interp.Finalize();

   SparseMatrix &I = interp.SpMat();
   // SparseMatrix I;
   // interp.FormRectangularSystemMatrix(h1_ess_dofs, constraints.bdr_dofs, I);

   SparseMatrix &P = constraints.GetProlongationMatrix();
   SparseMatrix &R = constraints.GetRestrictionMatrix();

   auto save_matrix = [](const SparseMatrix &A, const string &fname)
   {
      ofstream f(fname);
      A.PrintMatlab(f);
   };

   save_matrix(I, "I.txt");
   save_matrix(R, "R.txt");
   save_matrix(P, "P.txt");

   unique_ptr<SparseMatrix> RI(Mult(R, I));
   for (const int i : h1_ess_dofs) { RI->EliminateCol(i); }
   for (const int i : constraints.bdr_dofs) { RI->EliminateRow(i); }

   Array<int> empty;
   BilinearForm a(&rt_fes);
   a.AddDomainIntegrator(new VectorFEDiffusionIntegrator);
   a.Assemble();
   SparseMatrix A;
   a.FormSystemMatrix(empty, A);

   unique_ptr<SparseMatrix> A_c(RAP(P, A, P));
   unique_ptr<SparseMatrix> A_h1(RAP(I, A, I));

   save_matrix(*A_c, "A_c.txt");
   save_matrix(*A_h1, "A_h1.txt");

   // BCs...
   A_c->EliminateBC(constraints.bdr_dofs, SparseMatrix::DIAG_ONE);
   A_h1->EliminateBC(h1_ess_dofs, SparseMatrix::DIAG_ONE);

   // PatchSmoother S(constraints, *A_c);
   DSmoother S(*A_c);
   UMFPackSolver C(*A_h1);

   AdditiveSchwarz as(S, C, *RI);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A_c);
   cg.SetPreconditioner(as);

   Vector X(A_c->Height()), B(A_c->Height());

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
