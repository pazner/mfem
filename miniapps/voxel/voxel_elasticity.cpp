#include "mfem.hpp"
#include "mg_elasticity.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   string image_file = "australia.pgm";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&image_file, "-i", "--image", "Image file to use for mesh.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   const int dim = 2;
   H1_FECollection fec(order, dim);
   ImageElasticityMultigrid mg(image_file, fec);

   FiniteElementSpace &fespace = mg.GetFineSpace();
   Mesh &mesh = *fespace.GetMesh();

   Vector f_vec(dim);
   f_vec = 1.0;
   VectorConstantCoefficient f(f_vec);

   LinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(f));
   b.Assemble();

   GridFunction x(&fespace);
   x = 0.0;

   OperatorHandle A;
   Vector B, X;
   mg.FormFineLinearSystem(x, b, A, X, B);

   // CG(*A, B, X, 1, 200, 1e-12, 0.0);
   PCG(*A, mg, B, X, 1, 200, 1e-12, 0.0);

   // mg.GetFineForm().RecoverFEMSolution(X, b, x);
   // x.Save("sol.gf");

   mesh.Save("original.mesh");
   mesh.SetNodalFESpace(&fespace);
   *mesh.GetNodes() += x;
   mesh.Save("displaced.mesh");
   x *= -1;
   x.Save("sol.gf");
   x *= -1;

   ParaViewDataCollection pv("VoxelMG", fespace.GetMesh());
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   pv.SetLevelsOfDetail(order + 1);
   pv.RegisterField("u", &x);
   pv.SetCycle(0);
   pv.SetTime(0);
   pv.Save();

   Vector x_prev = x;

   const int nl = mg.nlevels;
   for (int i = 0; i < nl - 1; ++i)
   {
      FiniteElementSpace &space = *mg.spaces[nl-2-i];
      GridFunction x_c(&space);
      mg.prolongations[nl-2-i]->Coarsen(x_prev, x_c);
      x_prev = x_c;

      space.GetMesh()->SetNodalFESpace(&space);
      (*space.GetMesh()->GetNodes()) += x_c;

      pv.SetMesh(space.GetMesh());
      x_c *= -1.0;
      pv.RegisterField("u", &x_c);
      pv.SetCycle(i+1);
      pv.SetTime(i+1);
      pv.Save();
      x_c *= -1.0;
   }

   return 0;
}
