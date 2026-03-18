#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   vector<string> meshes =
   {
      "meshes/BL3.msh",
      "meshes/BL3_surf.msh",
      "meshes/naca0012_bl_o3.msh",
      "meshes/visc_BGM45-15_reg.msh"
   };

   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   ParaViewDataCollection pv("LORMesh", nullptr);
   pv.SetPrefixPath("meshes/ParaView");

   for (auto mesh_file : meshes)
   {
      cout << mesh_file << '\n';
      Mesh mesh(mesh_file, 0, 0, true);
      Mesh mesh_lor = Mesh::MakeDuffyRefined(mesh, order, BasisType::GaussLobatto);

      pv.SetHighOrderOutput(true);
      pv.SetLevelsOfDetail(3);
      pv.SetMesh(&mesh);
      pv.Save();
      pv.SetCycle(pv.GetCycle() + 1);
      pv.SetTime(pv.GetTime() + 1);

      pv.SetHighOrderOutput(false);
      pv.SetLevelsOfDetail(1);
      pv.SetMesh(&mesh_lor);
      pv.Save();
      pv.SetCycle(pv.GetCycle() + 1);
      pv.SetTime(pv.GetTime() + 1);
   }


   return 0;
}
