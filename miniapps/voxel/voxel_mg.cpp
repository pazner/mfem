#include "mfem.hpp"
#include "voxel_mesh.hpp"
#include "mg.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   string mesh_file =
      "/Users/pazner/Documents/portland_state/10_research/13_meshes/bone_72k.mesh";
   int order = 1;
   double h = 0.32;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&h, "-hx", "--hx", "Mesh cell size.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   H1_FECollection fec(order, 3);

   VoxelMultigrid mg(VoxelMesh(mesh_file), fec);
   FiniteElementSpace &fespace = mg.GetFineSpace();
   const int nl = mg.nlevels;

   for (int i = 0; i < nl; ++i)
   {
      std::cout << "Level " << i << "\\\\";
      std::cout << mg.meshes[nl-1-i]->GetNE() << " elements\\\\";
      std::cout << mg.spaces[nl-1-i]->GetTrueVSize() << " DOFs\n";
   }

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

#if 0

   VoxelMesh mesh(mesh_file, h);
   VoxelMesh coarsened_mesh = mesh.Coarsen();

   FiniteElementSpace fine_fes(&mesh, &fec);
   FiniteElementSpace coarse_fes(&coarsened_mesh, &fec);

   GridFunction fine_gf(&fine_fes);
   GridFunction coarse_gf(&coarse_fes);

   auto fn = [](const Vector &xvec)
   {
      const double x = xvec[0];
      const double y = xvec[1];
      return exp(0.1*sin(5.1*x - 6.2*y) + 0.3*cos(4.3*x +3.4*y));
   };
   FunctionCoefficient fn_coeff(fn);
   coarse_gf.ProjectCoefficient(fn_coeff);

   Array<int> empty;
   VoxelProlongation P(coarse_fes, empty, fine_fes, empty);
   P.Mult(coarse_gf, fine_gf);

   // coarsened_mesh.Save("coarsened_voxel_mesh.mesh");

   ParaViewDataCollection pv("VoxelMG", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   pv.SetLevelsOfDetail(order + 1);

   pv.RegisterField("u", &fine_gf);
   pv.SetCycle(0);
   pv.SetTime(0);
   pv.Save();

   pv.SetMesh(&coarsened_mesh);
   pv.RegisterField("u", &coarse_gf);
   pv.SetCycle(1);
   pv.SetTime(1);
   pv.Save();

#endif

   return 0;
}
