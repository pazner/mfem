#include "mfem.hpp"
#include "image_mesh.hpp"
#include "mg.hpp"

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

   H1_FECollection fec(order, 2);
   ImageMultigrid mg(image_file, fec);

   const int nl = mg.nlevels;

   for (int i = 0; i < nl; ++i)
   {
      std::cout << "Level " << i << "\\\\";
      std::cout << mg.meshes[nl-1-i]->GetNE() << " elements\\\\";
      std::cout << mg.spaces[nl-1-i]->GetTrueVSize() << " DOFs\n";
   }

   FiniteElementSpace &fespace = mg.GetFineSpace();

   ConstantCoefficient one(1.0);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   GridFunction x(&fespace);
   x = 0.0;

   OperatorHandle A;
   Vector B, X;
   mg.FormFineLinearSystem(x, b, A, X, B);

   PCG(*A, mg, B, X, 1, 200, 1e-12, 0.0);

   mg.GetFineForm().RecoverFEMSolution(X, b, x);
   x.Save("sol.gf");
   fespace.GetMesh()->Save("mesh.mesh");

   ParaViewDataCollection pv("VoxelMG", fespace.GetMesh());
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   pv.SetLevelsOfDetail(order + 1);
   pv.RegisterField("u", &x);
   pv.SetCycle(0);
   pv.SetTime(0);
   pv.Save();

   Vector x_prev = x;

   for (int i = 0; i < nl - 1; ++i)
   {
      FiniteElementSpace &space = *mg.spaces[nl-2-i];
      GridFunction x_c(&space);
      mg.prolongations[nl-2-i]->Coarsen(x_prev, x_c);
      x_prev = x_c;

      pv.SetMesh(space.GetMesh());
      pv.RegisterField("u", &x_c);
      pv.SetCycle(i+1);
      pv.SetTime(i+1);
      pv.Save();
   }

   return 0;
}
