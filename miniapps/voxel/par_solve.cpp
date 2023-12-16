#include "mfem.hpp"
#include "voxel_mesh.hpp"
#include "par_mg.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const string &prefix, int rank)
{
   ifstream f(MakeParFilename(prefix + ".mesh.", rank));
   return ParMesh(MPI_COMM_WORLD, f, false);
}

ParVoxelMapping LoadVoxelMapping(const string &prefix, int rank)
{
   ParVoxelMapping mapping;

   ifstream f(MakeParFilename(prefix + ".mapping.", rank));
   MFEM_VERIFY(f.good(), "Error opening ifstream");

   // Load local parents
   int local_parents_size;
   f >> local_parents_size;
   mapping.local_parents.SetSize(local_parents_size);
   for (int i = 0; i < local_parents_size; ++i)
   {
      f >> mapping.local_parents[i].element_index;
      f >> mapping.local_parents[i].pmat_index;
   }

   // Load local parent offsets
   int local_parent_offsets_size;
   f >> local_parent_offsets_size;
   mapping.local_parent_offsets.SetSize(local_parent_offsets_size);
   for (int i = 0; i < local_parent_offsets_size; ++i)
   {
      f >> mapping.local_parent_offsets[i];
   }

   // Read coarse to fine
   int coarse_to_fine_size;
   f >> coarse_to_fine_size;
   mapping.coarse_to_fine.resize(coarse_to_fine_size);
   for (int i = 0; i < coarse_to_fine_size; ++i)
   {
      f >> mapping.coarse_to_fine[i].rank;
      int c2f_size;
      f >> c2f_size;
      mapping.coarse_to_fine[i].coarse_to_fine.resize(c2f_size);
      for (int j = 0; j < c2f_size; ++j)
      {
         f >> mapping.coarse_to_fine[i].coarse_to_fine[j].coarse_element_index;
         f >> mapping.coarse_to_fine[i].coarse_to_fine[j].pmat_index;
      }
   }

   // Read fine to coarse
   int fine_to_coarse_size;
   f >> fine_to_coarse_size;
   mapping.fine_to_coarse.resize(fine_to_coarse_size);
   for (int i = 0; i < fine_to_coarse_size; ++i)
   {
      f >> mapping.fine_to_coarse[i].rank;
      int f2c_size;
      f >> f2c_size;
      mapping.fine_to_coarse[i].fine_to_coarse.resize(f2c_size);
      for (int j = 0; j < f2c_size; ++j)
      {
         f >> mapping.fine_to_coarse[i].fine_to_coarse[j].fine_element_index;
         f >> mapping.fine_to_coarse[i].fine_to_coarse[j].pmat_index;
      }
   }

   return mapping;
}

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

   const int rank = Mpi::WorldRank();

   ParVoxelMapping mapping = LoadVoxelMapping(dir + "/transfer", rank);
   ParMesh fine_mesh = LoadParMesh(dir + "/fine", rank);
   ParMesh coarse_mesh = LoadParMesh(dir + "/coarse", rank);

   H1_FECollection fec(order, fine_mesh.Dimension());
   ParFiniteElementSpace fine_fes(&fine_mesh, &fec);
   ParFiniteElementSpace coarse_fes(&coarse_mesh, &fec);

   Array<int> empty;

   ParVoxelProlongation P(coarse_fes, empty, fine_fes, empty, mapping);

   ParGridFunction xc(&coarse_fes);
   FunctionCoefficient coeff([](const Vector &xvec)
   {
      const double x = xvec[0];
      const double y = xvec[1];
      const double z = xvec[2];
      return exp(0.1*sin(5.1*x - 6.2*y + 2.7*z) + 0.3*cos(4.3*x + 3.4*y - 8.1*z));
   });
   xc.ProjectCoefficient(coeff);

   Vector xc_tv(coarse_fes.GetTrueVSize());
   xc.GetTrueDofs(xc_tv);

   Vector xf_tv(fine_fes.GetTrueVSize());
   P.Mult(xc_tv, xf_tv);

   ParGridFunction xf(&fine_fes);
   xf.SetFromTrueDofs(xf_tv);

   ParaViewDataCollection pv("ParSolve", &fine_mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   pv.SetLevelsOfDetail(order + 1);
   pv.RegisterField("u", &xf);
   pv.SetCycle(0);
   pv.SetTime(0);
   pv.Save();

   pv.SetMesh(&coarse_mesh);
   pv.RegisterField("u", &xc);
   pv.SetCycle(1);
   pv.SetTime(1);
   pv.Save();

   return 0;
}
