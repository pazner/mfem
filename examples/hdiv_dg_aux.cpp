#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "hdiv_prolongation.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

// Vector DG diffusion integrator
class VectorDGDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;
   MatrixCoefficient *MQ;
   double sigma, kappa;
   int vdim;

   // these are not thread-safe!
   Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
   DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
   VectorDGDiffusionIntegrator(const double s, const double k,
                               const int vdim_) : Q(NULL), MQ(NULL), sigma(s), kappa(k), vdim(vdim_) { }
   VectorDGDiffusionIntegrator(Coefficient &q, const double s, const double k,
                               const int vdim_) : Q(&q), MQ(NULL), sigma(s), kappa(k), vdim(vdim_) { }
   VectorDGDiffusionIntegrator(MatrixCoefficient &q, const double s,
                               const double k, const int vdim_) : Q(NULL), MQ(&q), sigma(s), kappa(k),
      vdim(vdim_) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix( const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &full_elmat);
};

struct AuxiliaryPreconditioner : Solver
{
   const Solver &B; // Auxiliary preconditioner
   const Operator &Pi; // Transfer operator
   const Array<int> &ess_dofs;
   mutable Vector z1, z2;

   AuxiliaryPreconditioner(const Solver &B_, const Operator &Pi_,
                           const Array<int> &ess_dofs_)
      : Solver(Pi_.Height()), B(B_), Pi(Pi_), ess_dofs(ess_dofs_)
   { }

   void SetOperator(const Operator &op) { }

   void Mult(const Vector &b, Vector &x) const
   {
      z1.SetSize(Pi.Width());
      z2.SetSize(Pi.Width());

      Pi.MultTranspose(b, z1);
      B.Mult(z1, z2);
      Pi.Mult(z2, x);

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

   L2_FECollection l2_fec(order, dim, b1);
   FiniteElementSpace l2_fes(&mesh, &l2_fec, dim);

   cout << "RT DOFs: " << rt_fes.GetTrueVSize() << '\n';
   cout << "L2 DOFs: " << l2_fes.GetTrueVSize() << '\n';

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> rt_ess_dofs;
   Array<int> l2_ess_dofs; // empty
   rt_fes.GetEssentialTrueDofs(ess_bdr, rt_ess_dofs);

   DiscreteLinearOperator interp(&l2_fes, &rt_fes);
   interp.AddDomainInterpolator(new IdentityInterpolator);
   interp.Assemble();
   interp.Finalize();

   SparseMatrix I;
   interp.FormRectangularSystemMatrix(l2_ess_dofs, rt_ess_dofs, I);

   BilinearForm a_l2(&l2_fes);
   a_l2.AddDomainIntegrator(new VectorDiffusionIntegrator);
   const double sigma = -1.0;
   a_l2.AddInteriorFaceIntegrator(new VectorDGDiffusionIntegrator(sigma, kappa,
                                                                  dim));
   a_l2.AddBdrFaceIntegrator(new VectorDGDiffusionIntegrator(sigma, kappa, dim));
   a_l2.Assemble();
   a_l2.Finalize();
   SparseMatrix &A_l2 = a_l2.SpMat();

   BilinearForm a(&rt_fes);
   a.AddDomainIntegrator(new VectorFEDiffusionIntegrator);
   a.AddInteriorFaceIntegrator(new VectorFE_DGDiffusionIntegrator(kappa));
   a.AddBdrFaceIntegrator(new VectorFE_DGDiffusionIntegrator(kappa));
   a.Assemble();
   a.SetDiagonalPolicy(Operator::DIAG_ONE);
   SparseMatrix A;
   a.FormSystemMatrix(rt_ess_dofs, A);

   auto save_matrix = [](const Operator &A, const string &fname)
   {
      ofstream f(fname);
      A.PrintMatlab(f);
   };

   save_matrix(A_l2, "A_l2.txt");
   save_matrix(A, "A.txt");
   save_matrix(I, "I.txt");

   // AdditiveSchwarz as(S, C, I);

   // save_matrix(as, "as.txt");

   UMFPackSolver A_l2_inv(A_l2);
   AuxiliaryPreconditioner aux(A_l2_inv, I, rt_ess_dofs);

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

void VectorDGDiffusionIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &full_elmat)
{
   int dim, ndof1, ndof2, ndofs;
   bool kappa_is_nonzero = (kappa != 0.);
   double w, wq = 0.0;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   dshape1dn.SetSize(ndof1);
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
      shape2.SetSize(ndof2);
      dshape2.SetSize(ndof2, dim);
      dshape2dn.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }

   ndofs = ndof1 + ndof2;
   DenseMatrix elmat;
   elmat.SetSize(ndofs);
   elmat = 0.0;
   if (kappa_is_nonzero)
   {
      jmat.SetSize(ndofs);
      jmat = 0.;
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order
      int order;
      if (ndof2)
      {
         order = 2 * max(el1.GetOrder(), el2.GetOrder());
      }
      else
      {
         order = 2 * el1.GetOrder();
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: < {(Q \nabla u).n},[v] >      --> elmat
   //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2 * eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      el1.CalcShape(eip1, shape1);
      el1.CalcDShape(eip1, dshape1);
      w = ip.weight / Trans.Elem1->Weight();
      if (ndof2)
      {
         w /= 2;
      }
      if (!MQ)
      {
         if (Q)
         {
            w *= Q->Eval(*Trans.Elem1, eip1);
         }
         ni.Set(w, nor);
      }
      else
      {
         nh.Set(w, nor);
         MQ->Eval(mq, *Trans.Elem1, eip1);
         mq.MultTranspose(nh, ni);
      }
      CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
      adjJ.Mult(ni, nh);
      if (kappa_is_nonzero)
      {
         wq = ni * nor;
      }
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //
      //      |nor|=measure(face)/measure(ref. face),
      //
      //      det(J1)=measure(element)/measure(ref. element),
      //
      //      and the ratios measure(ref. element)/measure(ref. face)
      //      are compatible for all element/face pairs.
      //
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      //
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      dshape1.Mult(nh, dshape1dn);
      for (int i = 0; i < ndof1; i++)
         for (int j = 0; j < ndof1; j++)
         {
            elmat(i, j) += shape1(i) * dshape1dn(j);
         }

      if (ndof2)
      {
         el2.CalcShape(eip2, shape2);
         el2.CalcDShape(eip2, dshape2);
         w = ip.weight / 2 / Trans.Elem2->Weight();
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(*Trans.Elem2, eip2);
            }
            ni.Set(w, nor);
         }
         else
         {
            nh.Set(w, nor);
            MQ->Eval(mq, *Trans.Elem2, eip2);
            mq.MultTranspose(nh, ni);
         }
         CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
         adjJ.Mult(ni, nh);
         if (kappa_is_nonzero)
         {
            wq += ni * nor;
         }

         dshape2.Mult(nh, dshape2dn);

         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(i, ndof1 + j) += shape1(i) * dshape2dn(j);
            }

         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(ndof1 + i, j) -= shape2(i) * dshape1dn(j);
            }

         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(ndof1 + i, ndof1 + j) -= shape2(i) * dshape2dn(j);
            }
      }

      if (kappa_is_nonzero)
      {
         // only assemble the lower triangular part of jmat
         wq *= kappa;
         for (int i = 0; i < ndof1; i++)
         {
            const double wsi = wq * shape1(i);
            for (int j = 0; j <= i; j++)
            {
               jmat(i, j) += wsi * shape1(j);
            }
         }
         if (ndof2)
         {
            for (int i = 0; i < ndof2; i++)
            {
               const int i2 = ndof1 + i;
               const double wsi = wq * shape2(i);
               for (int j = 0; j < ndof1; j++)
               {
                  jmat(i2, j) -= wsi * shape1(j);
               }
               for (int j = 0; j <= i; j++)
               {
                  jmat(i2, ndof1 + j) += wsi * shape2(j);
               }
            }
         }
      }
   }

   // elmat := -elmat + sigma*elmat^t + jmat
   if (kappa_is_nonzero)
   {
      for (int i = 0; i < ndofs; i++)
      {
         for (int j = 0; j < i; j++)
         {
            double aij = elmat(i, j), aji = elmat(j, i), mij = jmat(i, j);
            elmat(i, j) = sigma * aji - aij + mij;
            elmat(j, i) = sigma * aij - aji + mij;
         }
         elmat(i, i) = (sigma - 1.) * elmat(i, i) + jmat(i, i);
      }
   }
   else
   {
      for (int i = 0; i < ndofs; i++)
      {
         for (int j = 0; j < i; j++)
         {
            double aij = elmat(i, j), aji = elmat(j, i);
            elmat(i, j) = sigma * aji - aij;
            elmat(j, i) = sigma * aij - aji;
         }
         elmat(i, i) *= (sigma - 1.);
      }
   }

   // populate full matrix following github issue #2909
   full_elmat.SetSize(vdim*(ndof1 + ndof2));
   full_elmat = 0.0;
   for (int d=0; d<vdim; ++d)
   {
      for (int j=0; j<ndofs; ++j)
      {
         int jj = (j < ndof1) ? j + d*ndof1 : j - ndof1 + d*ndof2 + vdim*ndof1;
         for (int i=0; i<ndofs; ++i)
         {
            int ii = (i < ndof1) ? i + d*ndof1 : i - ndof1 + d*ndof2 + vdim*ndof1;
            full_elmat(ii, jj) += elmat(i, j);
         }
      }
   }
};
