#include "mfem.hpp"
#include "voxel_mesh.hpp"

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

   H1_FECollection fec(order, 2);
   VoxelMesh mesh(mesh_file, h);

   VoxelMesh coarsened_mesh = mesh.Coarsen();

   coarsened_mesh.Save("coarsened_voxel_mesh.mesh");

   ParaViewDataCollection pv("VoxelMG", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   pv.SetLevelsOfDetail(order + 1);

   pv.SetCycle(0);
   pv.SetTime(0);
   pv.Save();

   pv.SetMesh(&coarsened_mesh);
   pv.SetCycle(1);
   pv.SetTime(1);
   pv.Save();

   return 0;
}
