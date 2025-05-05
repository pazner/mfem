#include "mfem.hpp"
#include "voxel_mesh.hpp"
#include "par_mg.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   string dir = "VoxelData/Voxel";
   int order = 2;
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&dir, "-d", "--dir", "Data directory.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable ParaView output.");
   args.ParseCheck();

   MFEM_VERIFY(order >= 2, "Taylor-Hood requires velocity degree >= 2.");

   // List of boundary attributes that should be treated as essential (e.g.
   // Dirichlet or displacement boundary conditions.)
   std::vector<int> u_ess_bdr = { 1, 3 };

   ParVoxelMultigrid mg(dir, order, ProblemType::VectorPoisson, u_ess_bdr);
   ParFiniteElementSpace &u_fes = mg.GetFineSpace();
   ParMesh &mesh = *u_fes.GetParMesh();

   auto &ba = mesh.bdr_attribute_sets;
   ba.CreateAttributeSet("Essential Velocity");
   for (int attr : u_ess_bdr)
   {
      ba.AddToAttributeSet("Essential Velocity", attr);
   }
   ba.CreateAttributeSet("Essential Pressure");
   for (int attr : mesh.bdr_attributes)
   {
      if (ba.attr_sets["Essential Velocity"].Find(attr) == -1)
      {
         ba.AddToAttributeSet("Essential Pressure", attr);
      }
   }
   ba.CreateAttributeSet("Inflow");
   ba.AddToAttributeSet("Inflow", 3);
   auto u_bdr_is_ess = ba.GetAttributeSetMarker("Essential Velocity");
   auto p_bdr_is_ess = ba.GetAttributeSetMarker("Essential Pressure");
   auto bdr_is_inflow = ba.GetAttributeSetMarker("Inflow");

   H1_FECollection p_fec(order - 1, mesh.Dimension());
   ParFiniteElementSpace p_fes(&mesh, &p_fec);

   Array<int> p_ess_dofs;
   p_fes.GetEssentialTrueDofs(p_bdr_is_ess, p_ess_dofs);
   Array<int> u_ess_dofs;
   u_fes.GetEssentialTrueDofs(u_bdr_is_ess, u_ess_dofs);

   ParMixedBilinearForm d(&u_fes, &p_fes);
   d.AddDomainIntegrator(new VectorDivergenceIntegrator);
   d.Assemble();
   d.Finalize();
   unique_ptr<HypreParMatrix> D_full(d.ParallelAssemble()); // deep copy
   HypreParMatrix D;
   d.FormRectangularSystemMatrix(u_ess_dofs, p_ess_dofs, D);
   unique_ptr<HypreParMatrix> Dt(D.Transpose());

   ParMixedBilinearForm g(&p_fes, &u_fes);
   g.AddDomainIntegrator(new GradientIntegrator);
   g.Assemble();
   g.Finalize();
   HypreParMatrix G;
   g.FormRectangularSystemMatrix(p_ess_dofs, u_ess_dofs, G);

   ParBilinearForm w(&p_fes);
   w.AddDomainIntegrator(new MassIntegrator);
   w.Assemble();
   w.Finalize();
   HypreParMatrix W;
   w.FormSystemMatrix(p_ess_dofs, W);

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = u_fes.GetTrueVSize();
   offsets[2] = offsets[1] + p_fes.GetTrueVSize();

   BlockOperator A(offsets);
   A.SetBlock(0, 0, &mg.GetFineOperator());
   A.SetBlock(0, 1, &G);
   A.SetBlock(1, 0, &D, -1.0);

   HypreSmoother w_diag(W, HypreSmoother::Jacobi);

   BlockDiagonalPreconditioner prec(offsets);
   prec.SetDiagonalBlock(0, &mg);
   prec.SetDiagonalBlock(1, &w_diag);

   // Set inflow conditions
   Vector inflow_vec(mesh.Dimension());
   inflow_vec = 0.0;
   inflow_vec[0] = 1.0;
   VectorConstantCoefficient inflow_coeff(inflow_vec);

   ParGridFunction u(&u_fes);
   u = 0.0;
   u.ProjectBdrCoefficient(inflow_coeff, bdr_is_inflow);

   BlockVector X(offsets);
   X = 0.0;
   u.GetTrueDofs(X.GetBlock(0));

   Vector coeff_vector(mesh.Dimension());
   coeff_vector = 0.0;
   VectorConstantCoefficient f_coeff(coeff_vector);

   ParLinearForm f(&u_fes);
   f.AddDomainIntegrator(new VectorDomainLFIntegrator(f_coeff));
   f.Assemble();
   Vector F(u_fes.GetTrueVSize());
   f.ParallelAssemble(F);

   BlockVector B(offsets);
   B.GetBlock(0) = F;
   B.GetBlock(1) = 0.0;

   // Account for boundary conditions on the right-hand side
   {
      auto *c_A = dynamic_cast<ConstrainedOperator*>(&mg.GetFineOperator());
      c_A->EliminateRHS(X.GetBlock(0), B.GetBlock(0));

      Vector B_bdr_1(p_fes.GetTrueVSize());
      D_full->Mult(X.GetBlock(0), B_bdr_1);
      B.GetBlock(1) += B_bdr_1;

      B.GetBlock(1).SetSubVector(p_ess_dofs, 0.0);
   }

   MINRESSolver minres(MPI_COMM_WORLD);
   minres.SetRelTol(1e-8);
   minres.SetMaxIter(500);
   minres.SetOperator(A);
   minres.SetPreconditioner(prec);
   minres.SetPrintLevel(IterativeSolver::PrintLevel().Iterations());
   minres.Mult(B, X);

   u.SetFromTrueDofs(X.GetBlock(0));

   ParGridFunction p(&p_fes);
   p.SetFromTrueDofs(X.GetBlock(1));

   if (visualization)
   {
      unique_ptr<ParaViewDataCollectionBase> pv;

#ifdef MFEM_USE_HDF5
      pv.reset(new ParaViewHDFDataCollection("Stokes", &mesh));
#else
      pv.reset(new ParaViewDataCollection("Stokes", &mesh));
#endif

      pv->SetPrefixPath("ParaView");
      pv->SetHighOrderOutput(true);
      pv->SetLevelsOfDetail(order);
      pv->RegisterField("u", &u);
      pv->RegisterField("p", &p);
      pv->Save();
   }

   return 0;
}
