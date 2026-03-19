#include "mfem.hpp"
#include "fem/picojson.h"
#include "fem/integ/bilininteg_mass_kernels.hpp"
#include "fem/integ/bilininteg_diffusion_kernels.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class MatrixFreeDuffyOperator : public Operator
{
   int ne, d1d, q1d;
   Array<real_t> B, Bt, G, Gt;
   Vector pa_data_diff, pa_data_mass;

   const Operator *P;
   const Array<int> &ess_dofs;

   Array<int> gather;

   mutable Vector z;
   mutable Vector x_lvec, y_lvec;
   mutable Vector x_evec, y_evec;
public:
   MatrixFreeDuffyOperator(const FiniteElementSpace &fes, real_t alpha,
                           real_t beta, const Array<int> &ess_dofs_)
      : Operator(fes.GetTrueVSize()),
        ne(fes.GetNE()),
        P(fes.GetProlongationMatrix()),
        ess_dofs(ess_dofs_)
   {
      const int order = fes.GetMaxElementOrder();
      const int mesh_order = [&]()
      {
         if (auto nodal_fes = fes.GetMesh()->GetNodalFESpace())
         {
            return nodal_fes->GetMaxElementOrder() * 2 - 1;
         }
         else
         {
            return 1;
         }
      }();
      IntegrationRule ir = IntRules.Get(Geometry::SQUARE, 2*order + 3);
      const int nq = ir.Size();

      IntegrationRule ir_collapsed = ir;
      for (int iq = 0; iq < nq; ++iq)
      {
         ir_collapsed[iq].x *= 1.0 - ir_collapsed[iq].y;
         ir_collapsed[iq].weight *= 1.0 - ir_collapsed[iq].y;
      }

      d1d = order + 1;
      q1d = sqrt(nq);

      H1_QuadrilateralElement quad_fe(order, BasisType::GaussLobatto);
      auto &d2q = quad_fe.GetDofToQuad(ir, DofToQuad::TENSOR);

      B = d2q.B;
      Bt = d2q.Bt;
      G = d2q.G;
      Gt = d2q.Gt;

      pa_data_mass.SetSize(nq * ne);
      pa_data_diff.SetSize(3 * nq * ne);

      auto D = Reshape(pa_data_diff.Write(), nq, 3, ne);

      for (int e = 0; e < fes.GetNE(); ++e)
      {
         auto &T = *fes.GetElementTransformation(e);

         for (int iq = 0; iq < nq; ++iq)
         {
            const bool tri = T.GetGeometryType() == Geometry::TRIANGLE;
            IntegrationPoint ip;
            if (tri) { ip = ir_collapsed[iq]; }
            else { ip = ir[iq]; }

            T.SetIntPoint(&ip);
            const real_t detJ = T.Weight();
            const real_t w = ip.weight;
            pa_data_mass[iq + e*nq] = alpha * w * detJ;

            auto &J = T.Jacobian();

            const real_t x = ir[iq].x;
            const real_t f = 1.0 - ir[iq].y;

            const real_t J11 = tri ? f*J(0,0) : J(0,0);
            const real_t J21 = tri ? f*J(1,0) : J(1,0);
            const real_t J12 = tri ? J(0,1) - x*J(0,0) : J(0,1);
            const real_t J22 = tri ? J(1,1) - x*J(1,0) : J(1,1);
            const real_t w_detJ = ir[iq].weight / ((J11*J22)-(J21*J12));

            D(iq,0,e) =  w_detJ * beta * (J12*J12 + J22*J22); // 1,1
            D(iq,1,e) = -w_detJ * beta * (J12*J11 + J22*J21); // 1,2
            D(iq,2,e) =  w_detJ * beta * (J11*J11 + J21*J21); // 2,2
         }
      }

      const int nd_quad = quad_fe.GetDof();

      x_lvec.SetSize(fes.GetVSize());
      y_lvec.SetSize(fes.GetVSize());

      x_evec.SetSize(ne * nd_quad);
      y_evec.SetSize(ne * nd_quad);

      gather.SetSize(ne * nd_quad);

      Array<int> dofs;
      for (int e = 0; e < fes.GetNE(); ++e)
      {
         auto *fe = fes.GetFE(e);
         auto *n_fe = dynamic_cast<const NodalFiniteElement*>(fe);
         MFEM_ASSERT(n_fe, "");

         auto &lex = n_fe->GetLexicographicOrdering();

         fes.GetElementDofs(e, dofs);

         for (int d = 0; d < fe->GetDof(); ++d)
         {
            gather[d + nd_quad*e] = dofs[lex[d]];
         }
         if (fe->GetGeomType() == Geometry::TRIANGLE)
         {
            const int top = dofs[2];
            for (int d = fe->GetDof(); d < nd_quad; ++d)
            {
               gather[d + nd_quad*e] = top;
            }
         }
      }
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      z = x;
      for (int i : ess_dofs) { z[i] = 0.0; }

      if (P) { P->Mult(z, x_lvec); }
      else { x_lvec = z; }

      for (int i = 0; i < gather.Size(); ++i)
      {
         x_evec[i] = x_lvec[gather[i]];
      }

      y_evec = 0.0;
      DiffusionIntegrator::ApplyPAKernels::Run(
         2, d1d, q1d, ne, true, B, G, Bt, Gt, pa_data_diff, x_evec, y_evec, d1d, q1d);
      MassIntegrator::ApplyPAKernels::Run(
         2, d1d, q1d, ne, B, Bt, pa_data_mass, x_evec, y_evec, d1d, q1d);

      y_lvec = 0.0;
      for (int i = 0; i < gather.Size(); ++i)
      {
         y_lvec[gather[i]] += y_evec[i];
      }

      if (P) { P->MultTranspose(y_lvec, y); }
      else { y = y_lvec; }

      for (int i : ess_dofs) { y[i] = x[i]; }
   }
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 1. Parse command line options.
   string mesh_file = "../data/star.mesh";
   int order = 1;
   int ref = 0;
   bool duffy = false;
   bool lor = false;
   bool pa = false;
   bool vis = false;
   string solver_type = "mumps";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ref, "-r", "--refine", "Number of refinements");
   args.AddOption(&solver_type, "-s", "--solver", "Solver. MUMPS or AMG.");
   args.AddOption(&duffy, "-d", "--duffy", "-no-d", "--no-duffy", "Use Duffy?");
   args.AddOption(&lor, "-l", "--lor", "-no-l", "--no-lor", "Solver LOR system?");
   args.AddOption(&pa, "-pa", "--pa", "-no-pa", "--no-pa", "Partial assembly?");
   args.AddOption(&vis, "-v", "--vis", "-no-v", "--no-vis", "ParaView vis?");
   args.ParseCheck();

   ParMesh mesh = [&]()
   {
      Mesh serial_mesh(mesh_file, 0, 0, true);
      for (int i = 0; i < ref; ++i) { serial_mesh.UniformRefinement(); }
      return ParMesh(MPI_COMM_WORLD, serial_mesh);
   }();

   const int dim = mesh.Dimension();

   ParMesh mesh_lor = [&]()
   {
      if (duffy)
      {
         return ParMesh::MakeDuffyRefined(mesh, order, BasisType::GaussLobatto);
      }
      else
      {
         return ParMesh::MakeRefined(mesh, order, BasisType::GaussLobatto);
      }
   }();

   H1_FECollection fec_lor(1, dim);
   ParFiniteElementSpace fes_lor(&mesh_lor, &fec_lor);

   ParBilinearForm a_lor(&fes_lor);
   a_lor.SetDiagonalPolicy(Operator::DIAG_ONE);
   a_lor.AddDomainIntegrator(new DiffusionIntegrator);
   a_lor.AddDomainIntegrator(new MassIntegrator);
   a_lor.Assemble();

   Array<int> ess_dofs_lor;
   fes_lor.GetBoundaryTrueDofs(ess_dofs_lor);

   HypreParMatrix A_lor;
   a_lor.FormSystemMatrix(ess_dofs_lor, A_lor);

   unique_ptr<FiniteElementCollection> fec;
   if (duffy)
   {
      fec.reset(new H1Duffy_FECollection(order, dim));
   }
   else
   {
      fec.reset(new H1_FECollection(order, dim));
   }

   ParFiniteElementSpace fespace(&mesh, fec.get());
   const HYPRE_BigInt global_ndofs = fespace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << global_ndofs << endl;
   }

   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   ParGridFunction x(&fespace);
   x = 0.0;

   FunctionCoefficient coeff([](const Vector &coo)
   {
      return exp(0.1*sin(5.1*coo[0] - 6.2*coo[1]) + 0.3*cos(4.3*coo[0] +3.4*coo[1]));
   });

   ConstantCoefficient one(1.0);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   Operator *A = nullptr;

   ParBilinearForm a(&fespace);
   a.SetDiagonalPolicy(Operator::DIAG_ONE);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.AddDomainIntegrator(new MassIntegrator);

   HypreParMatrix A_mat;
   Vector B(fespace.GetTrueVSize()), X(fespace.GetTrueVSize());

   if (pa) { tic_toc.Restart(); }

   MatrixFreeDuffyOperator mf_op(fespace, 1.0, 1.0, boundary_dofs);

   if (!pa) { tic_toc.Restart(); }

   if (pa)
   {
      MFEM_VERIFY(duffy, "");
      A = &mf_op;
   }
   else
   {
      a.Assemble();
      a.FormLinearSystem(boundary_dofs, x, b, A_mat, X, B);
      A = &A_mat;
   }
   const real_t assemble_time = tic_toc.RealTime();

   B.Randomize(1);
   X = 0.0;

   unique_ptr<Solver> prec;
   if (solver_type == "amg" || solver_type == "ho-amg")
   {
      unique_ptr<HypreBoomerAMG> amg;
      if (solver_type == "amg")
      {
         amg = make_unique<HypreBoomerAMG>(A_lor);
      }
      else
      {
         MFEM_VERIFY(!pa, "");
         amg = make_unique<HypreBoomerAMG>(A_mat);
      }

      HYPRE_BoomerAMGSetCoarsenType(*amg, 6);
      HYPRE_BoomerAMGSetInterpType(*amg, 0);

      HYPRE_BoomerAMGSetCycleRelaxType(*amg, 89, 1);
      HYPRE_BoomerAMGSetCycleRelaxType(*amg, 89, 2);
      HYPRE_BoomerAMGSetCycleRelaxType(*amg, 9, 3);

      HYPRE_BoomerAMGSetNumSweeps(*amg, 2);

      HYPRE_BoomerAMGSetAggNumLevels(*amg, 0);

      HYPRE_BoomerAMGSetStrongThreshold(*amg, 0.5);

      amg->Setup(B, X);

      prec = std::move(amg);
   }
   else if (solver_type == "mumps")
   {
      prec.reset(new MUMPSSolver(A_lor));
   }
   else
   {
      MFEM_ABORT("Unknown solver type");
   }

   Vector Y(B.Size());
   tic_toc.Restart();
   for (int i = 0; i < 100; ++i)
   {
      A->Mult(B, Y);
   }
   const real_t elapsed_op_100 = tic_toc.RealTime();

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   if (lor)
   {
      cg.SetOperator(A_lor);
   }
   else
   {
      cg.SetOperator(*A);
   }
   cg.SetPreconditioner(*prec);


   tic_toc.Restart();
   cg.Mult(B, X);
   const double elapsed = tic_toc.RealTime();

   const int niter = cg.GetNumIterations();

   picojson::object results;
   results["niter"] = picojson::value(double(niter));
   results["ndofs"] = picojson::value(double(global_ndofs));
   results["solver"] = picojson::value(solver_type);
   results["N"] = picojson::value(double(order));
   results["mesh"] = picojson::value(mesh_file);
   results["ref"] = picojson::value(double(ref));
   if (!pa)
   {
      results["nnz"] = picojson::value(double(A_mat.NNZ()));
   }
   results["nnz_lor"] = picojson::value(double(A_lor.NNZ()));
   results["elapsed"] = picojson::value(elapsed);
   results["assemble_time"] = picojson::value(assemble_time);
   results["matvec_time"] = picojson::value(elapsed_op_100/100.0);
   results["pa"] = picojson::value(pa);


   HYPRE_MemoryPrintUsage(MPI_COMM_WORLD, 2, __FUNCTION__, __LINE__);
   // HYPRE_MemoryPrintUsage(MPI_COMM_WORLD, 1, __FUNCTION__, __LINE__);

   if (Mpi::Root())
   {
      cout << "=== " << picojson::value(results) << '\n';
   }

   if (vis)
   {
      a.RecoverFEMSolution(X, b, x);
      ParaViewDataCollection pv("PoiDuffy", &mesh);
      pv.SetPrefixPath("ParaView");
      pv.SetHighOrderOutput(true);
      pv.SetLevelsOfDetail(std::min(8, 2*order));
      pv.RegisterField("u", &x);
      pv.Save();
   }

   return 0;
}
