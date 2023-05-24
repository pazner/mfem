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

   PatchSmoother(const FiniteElementSpace &fes, const SparseMatrix &A,
                 const SparseMatrix &P,
                 const RT_ContinuityConstraints &constraints)
      : ess_dofs(constraints.bdr_dofs)
   {
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
      x *= 1.0/4.0;

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
         // smoother->SetOperatorSymmetry(true);
         PatchSmoother *smoother = new PatchSmoother(fes, *A0, *P, constraints);
         // UMFPackSolver *smoother = new UMFPackSolver(*A0);
         S.reset(smoother);
      }
   }
};

class H1_RT_Multigrid : public MultigridBase
{
public:
   vector<unique_ptr<H1_Intersect_RT>> h1rt;
   vector<unique_ptr<SparseMatrix>> prolongations;

   // Constructs a diffusion multigrid for the ParFiniteElementSpaceHierarchy
   // and the array of essential boundaries
   H1_RT_Multigrid(ParFiniteElementSpaceHierarchy &hierarchy, Array<int> &ess_bdr)
   {
      for (int level = 0; level < hierarchy.GetNumLevels(); ++level)
      {
         const bool coarse = level == 0;
         H1_Intersect_RT *op = new H1_Intersect_RT(hierarchy.GetFESpaceAtLevel(level),
                                                   ess_bdr, coarse);
         h1rt.emplace_back(op);
         AddLevel(op->A0.get(), op->S.get(), false, false);
      }

      for (int level = 0; level < hierarchy.GetNumLevels() - 1; ++level)
      {
         FiniteElementSpace &fes_coarse = hierarchy.GetFESpaceAtLevel(level);
         FiniteElementSpace &fes_fine = hierarchy.GetFESpaceAtLevel(level+1);

         OperatorHandle T_op(Operator::MFEM_SPARSEMAT);
         fes_fine.GetTransferOperator(fes_coarse, T_op);
         SparseMatrix *T = T_op.Is<SparseMatrix>();
         MFEM_VERIFY(T != nullptr, "");

         SparseMatrix &P = h1rt[level]->constraints.GetProlongationMatrix();
         SparseMatrix &R = h1rt[level+1]->constraints.GetRestrictionMatrix();

         unique_ptr<SparseMatrix> Rt(Transpose(R));

         prolongations.emplace_back(RAP(*Rt, *T, P));

         for (const int i : h1rt[level]->constraints.bdr_dofs)
         {
            prolongations.back()->EliminateCol(i);
         }

         // Test something:
         // SparseMatrix &P2 = h1rt[level+1]->constraints.GetProlongationMatrix();

         // Vector x(P.Width());
         // x.Randomize(1);

         // Vector y1(P.Height());
         // Vector y2(T->Height());

         // Vector z1(prolongations.back()->Height());
         // Vector z2(P2.Height());

         // P.Mult(x, y1);
         // T->Mult(y1, y2);

         // prolongations.back()->Mult(x, z1);
         // P2.Mult(z1, z2);

         // y2 -= z2;
         // std::cout << "Norm difference: " << y2.Normlinf() << '\n';
      }
   }

   const Operator *GetProlongationAtLevel(int level) const override
   {
      return prolongations[level].get();
   }
};

class MultigridHierarchy
{
   ParFiniteElementSpace coarse_space;
   ParFiniteElementSpaceHierarchy space_hierarchy;
public:
   MultigridHierarchy(ParMesh &coarse_mesh, FiniteElementCollection &fec,
                      int h_refs)
      : coarse_space(&coarse_mesh, &fec),
        space_hierarchy(&coarse_mesh, &coarse_space, false, false)
   {
      for (int level = 0; level < h_refs; ++level)
      {
         space_hierarchy.AddUniformlyRefinedLevel();
      }
   }

   ParFiniteElementSpace &GetFinestSpace()
   {
      return space_hierarchy.GetFinestFESpace();
   }

   ParMesh &GetFinestMesh()
   {
      return *space_hierarchy.GetFinestFESpace().GetParMesh();
   }

   ParFiniteElementSpaceHierarchy &GetSpaceHierarchy() { return space_hierarchy; }
};

struct AdditiveSchwarz : Solver
{
   H1_RT_Multigrid &mg;
   mutable Vector z1, z2, z3;

   AdditiveSchwarz(H1_RT_Multigrid &mg_) : mg(mg_) { }

   void SetOperator(const Operator &op) { }

   void Mult(const Vector &b, Vector &x) const
   {
      const Operator &P = *mg.GetProlongationAtLevel(0);
      z1.SetSize(P.Width());
      z2.SetSize(P.Width());
      z3.SetSize(P.Height());

      P.MultTranspose(b, z1);
      mg.GetSmootherAtLevel(0)->Mult(z1, z2);
      P.Mult(z2, z3);

      mg.GetSmootherAtLevel(1)->Mult(b, x);
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

   ParMesh coarse_mesh = LoadParMesh(mesh_file, ser_ref, par_ref);

   const int dim = coarse_mesh.Dimension();
   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLobatto;
   RT_FECollection fec(order-1, dim, b1, b2);

   MultigridHierarchy mg_hierarchy(coarse_mesh, fec, h_ref);
   ParMesh &mesh = mg_hierarchy.GetFinestMesh();

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   H1_RT_Multigrid mg(mg_hierarchy.GetSpaceHierarchy(), ess_bdr);

   {
      ParFiniteElementSpaceHierarchy &hierarchy = mg_hierarchy.GetSpaceHierarchy();
      ParFiniteElementSpace &fes_0 = hierarchy.GetFESpaceAtLevel(0);
      ParFiniteElementSpace &fes_1 = hierarchy.GetFESpaceAtLevel(1);

      ParMesh &mesh_0 = *fes_0.GetParMesh();
      ParMesh &mesh_1 = *fes_1.GetParMesh();

      ParGridFunction x_0(&fes_0);
      ParGridFunction x_1(&fes_1);

      VectorFunctionCoefficient coeff(2, [](const Vector &xvec, Vector &v)
      {
         v[0] = xvec[0] - 0.5*xvec[1];
         v[1] = 2*xvec[0] + 3*xvec[1];
      });
      x_0.ProjectCoefficient(coeff);

      SparseMatrix &P_0 = mg.h1rt[0]->constraints.GetProlongationMatrix();
      SparseMatrix &R_0 = mg.h1rt[0]->constraints.GetRestrictionMatrix();
      Vector X_0(R_0.Height());
      X_0.Randomize(3);
      P_0.Mult(X_0, x_0);

      ParaViewDataCollection pv("RT_GMG", &mesh_0);
      pv.SetPrefixPath("ParaView");
      pv.SetHighOrderOutput(true);
      pv.SetLevelsOfDetail(order + 1);
      pv.RegisterField("u", &x_0);
      pv.SetCycle(0);
      pv.SetTime(0.0);
      pv.Save();

      R_0.Mult(x_0, X_0);

      SparseMatrix &P_1 = mg.h1rt[1]->constraints.GetProlongationMatrix();
      Vector X_1(P_1.Width());
      mg.GetProlongationAtLevel(0)->Mult(X_0, X_1);
      P_1.Mult(X_1, x_1);

      pv.SetMesh(&mesh_1);
      pv.RegisterField("u", &x_1);
      pv.SetCycle(1);
      pv.SetTime(1.0);
      pv.Save();
   }

   {
      std::ofstream f("A0_mg.txt");
      mg.h1rt[0]->A0->PrintMatlab(f);
   }
   {
      std::ofstream f("A1_mg.txt");
      mg.h1rt[1]->A0->PrintMatlab(f);
   }
   if (mg.h1rt.size() > 1)
   {
      ofstream f("P_mg");
      mg.prolongations[0]->PrintMatlab(f);
   }
   {
      ofstream f("dofs_0.txt");
      mg.h1rt[0]->constraints.bdr_dofs.Print(f, 1);
   }
   {
      ofstream f("dofs_1.txt");
      mg.h1rt[1]->constraints.bdr_dofs.Print(f, 1);
   }

   Operator &A = *mg.GetOperatorAtFinestLevel();

   AdditiveSchwarz as(mg);

   Vector X(A.Width()), B(A.Width());
   B.Randomize(1);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(A);
   cg.SetPreconditioner(mg);
   // cg.SetPreconditioner(as);

   X = 0.0;
   cg.Mult(B, X);

   // B = 0.0;
   // X.Randomize(1);

   // Vector R(B.Size());
   // Vector Z(B.Size());
   // Vector E(B.Size());

   // ParGridFunction r(&mg_hierarchy.GetFinestSpace());
   // SparseMatrix &P_1 = mg.h1rt[1]->constraints.GetProlongationMatrix();

   // ParaViewDataCollection pv("AdditiveSchwarz", &mesh);
   // pv.SetPrefixPath("ParaView");
   // pv.SetHighOrderOutput(true);
   // pv.SetLevelsOfDetail(order + 1);
   // pv.RegisterField("r", &r);

   // r = 0.0;

   // for (const int j : mg.h1rt[1]->constraints.bdr_dofs)
   // {
   //    X[j] = 0.0;
   // }

   // for (int i = 0; i < 50; ++i)
   // {
   //    // Compute residual
   //    R = B;
   //    A.Mult(X, Z);
   //    R -= Z;

   //    std::cout << "Resnorm = " << R.Norml2() << '\n';

   //    P_1.Mult(R, r);
   //    pv.SetCycle(i);
   //    pv.SetTime(i);
   //    pv.Save();

   //    // Compute correciton
   //    as.Mult(R, E);
   //    // Add (damped) correction
   //    E *= (1.0/6.0);
   //    // E *= (1.0/12.0);
   //    X += E;

   //    // ess bcs
   //    for (const int j : mg.h1rt[1]->constraints.bdr_dofs)
   //    {
   //       X[j] = 0.0;
   //    }
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
