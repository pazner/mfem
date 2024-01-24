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

   // List of boundary attributes that should be treated as essential (e.g.
   // Dirichlet or displacement boundary conditions.)
   std::vector<int> ess_bdr_attrs = {1};

   ParVoxelMultigrid mg(dir, order, pt, ess_bdr_attrs);
   ParFiniteElementSpace &fespace = mg.GetFineSpace();
   ParMesh &mesh = *fespace.GetParMesh();

   ParLinearForm b(&fespace);
   if (pt == ProblemType::Poisson)
   {
      ConstantCoefficient coeff(1.0);
      b.AddDomainIntegrator(new DomainLFIntegrator(coeff));
      b.Assemble();
   }
   else
   {
      const int dim = mesh.Dimension();
      VectorArrayCoefficient force_coeff(dim);

      // force_coeff is a coefficient representing the vector field F = (F0, F1,
      // F2). On the entire domain, F0 = F1 = 0. On boundary attribute 2, we set
      // F2 = -235, representing a pull-down force. On the rest of the
      // boundaries, F2 = 0, representing traction-free boundary conditions.
      for (int i = 0; i < dim-1; i++)
      {
         force_coeff.Set(i, new ConstantCoefficient(0.0));
      }
      const int top_bdr_attr = 2;
      Vector pull_force(mesh.bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(top_bdr_attr - 1) = -235.0;
      force_coeff.Set(dim-1, new PWConstCoefficient(pull_force));

      b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(force_coeff));
      b.Assemble();
   }

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
      pv.SetLevelsOfDetail(order);
      pv.RegisterField("u", &x);
      pv.SetCycle(0);
      pv.SetTime(0);
      pv.Save();
   }

   return 0;
}
