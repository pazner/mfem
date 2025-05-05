#include "mfem.hpp"
#include "voxel_mesh.hpp"
#include "par_mg.hpp"
#include "../../fem/picojson.h"

#include <sys/stat.h>  // mkdir

using namespace std;
using namespace mfem;

void SavePartitionedMesh(const string &prefix, Mesh &mesh, int np,
                         int *partitioning)
{
   MeshPartitioner partitioner(mesh, np, partitioning);
   MeshPart mesh_part;
   for (int i = 0; i < np; i++)
   {
      partitioner.ExtractPart(i, mesh_part);

      ofstream f(MakeParFilename(prefix + ".mesh.", i));
      f.precision(16);
      mesh_part.Print(f);
   }
}

void SaveParVoxelMappings(const string &prefix,
                          const vector<ParVoxelMapping> &mappings)
{
   const int np = mappings.size();
   for (int i = 0; i < np; ++i)
   {
      ofstream f(MakeParFilename(prefix + ".mapping.", i));

      f << mappings[i].local_parents.Size() << '\n';
      for (const auto &p : mappings[i].local_parents)
      {
         f << p.element_index << '\n' << p.pmat_index << '\n';
      }
      f << '\n';

      f << mappings[i].local_parent_offsets.Size() << '\n';
      for (const int x : mappings[i].local_parent_offsets) { f << x << '\n'; }
      f << '\n';

      f << mappings[i].coarse_to_fine.size() << '\n';
      for (const auto &c2f : mappings[i].coarse_to_fine)
      {
         f << c2f.rank << '\n';
         f << c2f.coarse_to_fine.size() << '\n';
         for (const auto &x : c2f.coarse_to_fine)
         {
            f << x.coarse_element_index << '\n' << x.pmat_index << '\n';
         }
      }
      f << '\n';

      f << mappings[i].fine_to_coarse.size() << '\n';
      for (const auto &f2c : mappings[i].fine_to_coarse)
      {
         f << f2c.rank << '\n';
         f << f2c.fine_to_coarse.size() << '\n';
         for (const auto &x : f2c.fine_to_coarse)
         {
            f << x.fine_element_index << '\n' << x.pmat_index << '\n';
         }
      }
   }
}

int main(int argc, char *argv[])
{
   string mesh_file =
      "/Users/pazner/Documents/portland_state/10_research/13_meshes/bone_72k.mesh";
   string dir = "Voxel";
   int np = 1;
   int ref = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&np, "-np", "--n-partitions", "Number of mesh partitions.");
   args.AddOption(&dir, "-d", "--dir", "Data directory.");
   args.AddOption(&ref, "-r", "--refine", "Number of refinements.");
   args.ParseCheck();

   mkdir("VoxelData", 0777);
   dir = "VoxelData/" + dir;

   int err_flag = mkdir(dir.c_str(), 0777);
   err_flag = (err_flag && (errno != EEXIST)) ? 1 : 0;
   MFEM_VERIFY(err_flag == 0, "Could not create directory " << dir)

   cout << "\n";
   tic_toc.Restart();
   cout << "Reading fine mesh... " << flush;
   // Read the (fine) mesh from a file

   auto get_refined_mesh = [&]()
   {
      Mesh orig_mesh(mesh_file);
      for (int i = 0; i < ref; ++i)
      {
         orig_mesh.UniformRefinement();
      }
      return orig_mesh;
   };

   unique_ptr<VoxelMesh> mesh(new VoxelMesh(get_refined_mesh()));
   cout << "Done. " << tic_toc.RealTime() << endl;
   const int dim = mesh->Dimension();

   // Partition the fine mesh
   tic_toc.Restart();
   cout << "Partitioning... " << flush;
   Array<int> partitioning(mesh->GeneratePartitioning(np), mesh->GetNE(), true);
   cout << "Done. " << tic_toc.RealTime() << endl;

   cout << "\n";
   cout << "Level      Elements     Save Mesh      Coarsening      Hierarchy      Mapping"
        << '\n'
        << string(77, '=')
        << endl;

   cout << left << setprecision(5) << fixed;

   int level = 0;
   while (true)
   {
      cout << setw(11) << level << flush;
      cout << setw(13) << mesh->GetNE() << flush;

      tic_toc.Restart();
      const string level_str = dir + "/level_" + to_string(level);
      SavePartitionedMesh(level_str, *mesh, np, partitioning);
      mesh->PrintBdrVTU(dir + "/bdr_level_" + to_string(level));
      cout << setw(15) << tic_toc.RealTime() << flush;

      const vector<int> &bounds = mesh->GetVoxelBounds();
      if (!all_of(bounds.begin(), bounds.end(), [](int x) { return x >= 4; }))
      {
         break;
      }

      tic_toc.Restart();
      unique_ptr<VoxelMesh> new_mesh(new VoxelMesh(mesh->Coarsen()));
      cout << setw(16) << tic_toc.RealTime() << flush;

      // Get hierarchy information for the new level
      tic_toc.Restart();
      Array<ParentIndex> parents;
      Array<int> parent_offsets;
      GetVoxelParents(*new_mesh, *mesh, parents, parent_offsets);
      // Coarsen the partitioning
      Array<int> new_partitioning(new_mesh->GetNE());
      for (int i = 0; i < new_mesh->GetNE(); ++i)
      {
         const int parent_index = parents[parent_offsets[i]].element_index;
         new_partitioning[i] = partitioning[parent_index];
      }
      cout << setw(15) << tic_toc.RealTime() << flush;

      // Create and save the parallel mappings
      tic_toc.Restart();
      auto mappings = CreateParVoxelMappings(
                         np, dim, parents, parent_offsets, partitioning,
                         new_partitioning);
      SaveParVoxelMappings(level_str, mappings);
      cout << setw(16) << tic_toc.RealTime() << endl;

      swap(new_mesh, mesh);
      Swap(new_partitioning, partitioning);

      ++level;
   }

   cout << endl;

   {
      picojson::object info;
      info["np"] = picojson::value(double(np));
      info["nlevels"] = picojson::value(double(level + 1));
      ofstream f(dir + "/info.json");
      f << picojson::value(info) << '\n';
   }

   return 0;
}
