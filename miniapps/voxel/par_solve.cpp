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

   OptionsParser args(argc, argv);
   args.AddOption(&dir, "-d", "--dir", "Data directory.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   ParVoxelMultigrid mg(dir, order);
   ParFiniteElementSpace &fespace = mg.GetFineSpace();
   ParMesh &mesh = *fespace.GetParMesh();

   ConstantCoefficient one(1.0);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   GridFunction x(&fespace);
   x = 0.0;

   OperatorHandle A;
   Vector B, X;
   mg.FormFineLinearSystem(x, b, A, X, B);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.SetOperator(mg.GetFineOperator());
   cg.SetPreconditioner(mg);
   cg.Mult(B, X);

   mg.GetFineForm().RecoverFEMSolution(X, b, x);

   ParaViewDataCollection pv("ParVoxelMG", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   pv.SetLevelsOfDetail(order + 1);
   pv.RegisterField("u", &x);
   pv.SetCycle(0);
   pv.SetTime(0);
   pv.Save();

   return 0;
}
