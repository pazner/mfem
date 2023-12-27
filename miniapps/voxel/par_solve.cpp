#include "mfem.hpp"
#include "voxel_mesh.hpp"
#include "par_mg.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   string dir = "Voxel";
   int order = 1;
   string prob_str = "poisson";
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&dir, "-d", "--dir", "Data directory.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree.");
   args.AddOption(&prob_str, "-p", "--problem",
                  "Problem type {p,poisson} or {e,elasticity}.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable ParaView output.");
   args.ParseCheck();

   ProblemType pt = [prob_str]()
   {
      if (prob_str == "p" || prob_str == "poisson")
      {
         return ProblemType::Poisson;
      }
      else if (prob_str == "e" || prob_str == "elasticity")
      {
         return ProblemType::Elasticity;
      }
      else
      {
         if (Mpi::Root()) { cerr << "Invalid problem type.\n"; }
         std::exit(1);
      }
   }();

   ParVoxelMultigrid mg(dir, order, pt);
   ParFiniteElementSpace &fespace = mg.GetFineSpace();
   ParMesh &mesh = *fespace.GetParMesh();

   Vector f_vec(fespace.GetVDim());
   f_vec = 1.0;
   VectorConstantCoefficient f(f_vec);

   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(f));
   b.Assemble();

   ParGridFunction x(&fespace);
   x = 0.0;

   OperatorHandle A;
   Vector B, X;
   mg.FormFineLinearSystem(x, b, A, X, B);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-6);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(1);
   cg.SetOperator(mg.GetFineOperator());
   cg.SetPreconditioner(mg);
   cg.Mult(B, X);

   mg.GetFineForm().RecoverFEMSolution(X, b, x);

   if (visualization)
   {
      ParaViewDataCollection pv("ParVoxelMG", &mesh);
      pv.SetPrefixPath("ParaView");
      pv.SetHighOrderOutput(true);
      pv.SetLevelsOfDetail(order + 1);
      pv.RegisterField("u", &x);
      pv.SetCycle(0);
      pv.SetTime(0);
      pv.Save();
   }

   return 0;
}
