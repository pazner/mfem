#include "mfem.hpp"
#include "voxel_mesh.hpp"
#include "par_mg.hpp"

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
   double h = 0.32;
   int np = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&h, "-hx", "--hx", "Mesh cell size.");
   args.AddOption(&np, "-np", "--n-partitions", "Number of mesh partitions.");
   args.AddOption(&dir, "-d", "--dir", "Data directory.");
   args.ParseCheck();

   // Read the (fine) mesh from a file
   VoxelMesh mesh(mesh_file, h);

   // Create a partitioning
   Array<int> partitioning(mesh.GeneratePartitioning(np), mesh.GetNE(), true);

   // Coarsen once
   VoxelMesh coarse_mesh = mesh.Coarsen();
   // Coarsen the partitioning
   Array<ParentIndex> parents;
   Array<int> parent_offsets;

   GetVoxelParents(coarse_mesh, mesh, parents, parent_offsets);
   Array<int> coarse_partitioning(coarse_mesh.GetNE());
   for (int i = 0; i < coarse_mesh.GetNE(); ++i)
   {
      const int parent_index = parents[parent_offsets[i]].element_index;
      coarse_partitioning[i] = partitioning[parent_index];
   }

   SavePartitionedMesh(dir + "/fine", mesh, np, partitioning);
   SavePartitionedMesh(dir + "/coarse", coarse_mesh, np, coarse_partitioning);

   auto mappings = CreateParVoxelMappings(
                      np, mesh.Dimension(), parents, parent_offsets, partitioning,
                      coarse_partitioning);

   SaveParVoxelMappings(dir + "/transfer", mappings);

   return 0;
}
