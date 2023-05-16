#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

struct Entity
{
   const Mesh &mesh;
   const int dim;
   const int entity_index;

   Entity(const Mesh &mesh_, const int dim_, const int entity_index_)
      : mesh(mesh_), dim(dim_), entity_index(entity_index_) { }

   Geometry::Type GetGeometry() const
   {
      switch (dim)
      {
         case 0: return Geometry::POINT;
         case 1: return Geometry::SEGMENT;
         case 2: return Geometry::SQUARE;
         case 3: return Geometry::CUBE;
         default: MFEM_ABORT("Invalid entity dimension.")
      }
      return Geometry::INVALID;
   }

   int GetLocalIndex(const int element_index) const
   {
      const int mesh_dim = mesh.Dimension();
      Array<int> indices, orientations;

      if (dim == 0)
      {
         mesh.GetElementVertices(element_index, indices);
      }
      else if (dim == mesh_dim)
      {
         return 0;
      }
      else if (dim == 1)
      {
         mesh.GetElementEdges(element_index, indices, orientations);
      }
      else if (dim == 2)
      {
         mesh.GetElementFaces(element_index, indices, orientations);
      }

      auto found = std::find(indices.begin(), indices.end(), entity_index);
      MFEM_VERIFY(found != indices.end(), "Did not find index.");
      return std::distance(indices.begin(), found);
   }
};

struct ElementEntityDofs
{
   const Entity &entity;
   const int element_index; /// Index of the parent element
   const int local_entity_index; /// Index of the entity within the element
   Array<int> local_dofs;
   Array<int> global_dofs;

   // ElementEntityDofs(const Entity &entity_) : entity(entity_) { }

   ElementEntityDofs(const FiniteElementSpace &fes, const Entity &entity_,
                     int element_index_)
      : entity(entity_),
        element_index(element_index_),
        local_entity_index(entity.GetLocalIndex(element_index))
   {
      Array<int> element_dofs;
      fes.GetElementDofs(element_index, element_dofs);

      const Mesh &mesh = *fes.GetMesh();
      const int dim = mesh.Dimension();

      const FiniteElement *fe = fes.GetFE(element_index);
      auto tfe = dynamic_cast<const TensorBasisElement*>(fe);
      MFEM_VERIFY(tfe != NULL, "Only tensor elements supported.");
      const auto dof_map = tfe->GetDofMap();

      const int p = fes.GetElementOrder(element_index);
      const int loc = local_entity_index;

      // Set the local DOFs
      MFEM_VERIFY(dim == 2, "Not yet implemented.");
      if (entity.dim == 0)
      {
         // 3 -- 2
         // |    |
         // 0 -- 1
         local_dofs.SetSize(2);
         const int x_offset = (loc == 0 || loc == 3) ? 0 : 1;
         const int y_offset = (loc == 0 || loc == 1) ? 0 : 1;

         const int ndof_per_dim = p*(p+1);

         local_dofs[0] = x_offset*p + y_offset*p*(p-1);
         local_dofs[1] = ndof_per_dim + x_offset*(p-1) + y_offset*p*p;
      }
      else if (entity.dim == 1)
      {
         // * - 2 - *
         // |       |
         // 3       1
         // |       |
         // * - 0 - *
         local_dofs.SetSize(2*p + 1);

         // direction of edge (parallel to x or y axis?)
         const int dir = (loc == 0 || loc == 2) ? 0 : 1;

         const int stride_x = (dir == 0) ? 1 : p + 1;
         const int stride_y = (dir == 0) ? 1 : p;

         // Subtract two since excluding the vertices
         const int nx = ((dir == 0) ? p + 1 : p) - 2;
         const int ny = ((dir == 0) ? p : p + 1) - 2;

         local_dofs.SetSize(nx + ny);

         const int offset_x = (loc == 1) ? p : (loc == 2) ? (p+1)*(p-1) : 0;

         for (int ix = 0; ix < nx; ++ ix)
         {
            // (ix + 1) below since skipping the first DOF (vertex)
            local_dofs[ix] = offset_x + (ix + 1)*stride_x;
         }

         const int ndof_per_dim = p*(p+1);

         const int offset_y = (loc == 1) ? p - 1 : (loc == 2) ? p*p : 0;
         for (int iy = 0; iy < ny; ++iy)
         {
            // (iy + 1) below since skipping the first DOF (vertex)
            local_dofs[nx + iy] = ndof_per_dim + offset_y + (iy + 1)*stride_y;
         }
      }
      else if (entity.dim == 2)
      {
         // Exclude edges --- only include interior element DOFs
         local_dofs.SetSize(2*(p-1)*(p-2));
      }

      global_dofs.SetSize(local_dofs.Size());
      for (int i = 0; i < local_dofs.Size(); ++i)
      {
         const int sd = dof_map[local_dofs[i]];
         const int d = (sd >= 0) ? sd : -1 - sd;
         global_dofs[i] = element_dofs[d];
      }
   }
};

struct PrimaryElementEntity
{
   const Entity &entity;
   const ElementEntityDofs dofs;
   const int dof_offset;

   PrimaryElementEntity(const FiniteElementSpace &fes, Entity &entity_,
                        int element_index, int dof_offset_)
      : entity(entity_),
        dofs(fes, entity, element_index), dof_offset(dof_offset_)
   { }
};

struct ConstrainedElementEntity
{
   const Entity &entity;
   const PrimaryElementEntity &primary;
   ElementEntityDofs dofs;
   DenseMatrix mapping;

   ConstrainedElementEntity(const FiniteElementSpace &fes,
                            const PrimaryElementEntity &primary_, int element_index)
      : entity(primary_.entity),
        primary(primary_),
        dofs(fes, entity, element_index)
   {
      // Create the mapping matrix.
   }
};

struct EntityDofs
{
   Entity entity;
   PrimaryElementEntity primary;
   std::vector<ConstrainedElementEntity> constrained;

   EntityDofs(const FiniteElementSpace &fes, int entity_index, int entity_dim,
              const Table &el_table, int dof_offset)
      : entity(*fes.GetMesh(), entity_dim, entity_index),
        primary(fes, entity, GetPrimaryIndex(el_table, entity_index), dof_offset)
   {
      Array<int> row;
      el_table.GetRow(entity_index, row);
      for (int i = 1; i < row.Size(); ++i)
      {
         constrained.emplace_back(fes, primary, row[i]);
      }
   }

   // Version of the constructor for elements
   EntityDofs(const FiniteElementSpace &fes, int element_index, int dof_offset)
      : entity(*fes.GetMesh(), fes.GetMesh()->Dimension(), element_index),
        primary(fes, entity, element_index, dof_offset)
   { }

   int GetNPrimaryDofs() const
   {
      return primary.dofs.local_dofs.Size();
   }

private:
   static int GetPrimaryIndex(const Table &table, int index)
   {
      Array<int> row;
      table.GetRow(index, row);
      MFEM_VERIFY(row.Size() > 0, "");
      return row[0];
   }
};

class RT_ContinuityConstraints
{
public: // temporary
   std::vector<EntityDofs> entities;
   int n_primary_dofs;
   FiniteElementSpace &fes;
public:
   RT_ContinuityConstraints(FiniteElementSpace &fes_) : fes(fes_)
   {
      MFEM_VERIFY(fes.GetMaxElementOrder() >= 2, "Order must be at least 2.");
      Mesh &mesh = *fes.GetMesh();
      const int dim = mesh.Dimension();
      int nentities = mesh.GetNV() + mesh.GetNE();
      if (dim >= 2) { nentities += mesh.GetNEdges(); }
      if (dim == 3) { nentities += mesh.GetNFaces(); }
      entities.reserve(nentities);

      MFEM_VERIFY(dim == 2, "3D not yet implemented.");

      n_primary_dofs = 0;

      std::unique_ptr<Table> el_table(mesh.GetVertexToElementTable());
      for (int iv = 0; iv < mesh.GetNV(); ++iv)
      {
         entities.emplace_back(fes, iv, 0, *el_table, n_primary_dofs);
         n_primary_dofs += entities.back().GetNPrimaryDofs();
      }

      el_table.reset(Transpose(mesh.ElementToEdgeTable()));
      for (int ie = 0; ie < mesh.GetNEdges(); ++ie)
      {
         entities.emplace_back(fes, ie, 1, *el_table, n_primary_dofs);
         n_primary_dofs += entities.back().GetNPrimaryDofs();
      }

      for (int iel = 0; iel < mesh.GetNE(); ++iel)
      {
         entities.emplace_back(fes, iel, n_primary_dofs);
         n_primary_dofs += entities.back().GetNPrimaryDofs();
      }
   }
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../data/star.mesh";
   int order = 2;
   int ser_ref = 0;
   int par_ref = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.ParseCheck();

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);

   const int dim = mesh.Dimension();
   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLobatto;
   RT_FECollection fec(order-1, dim, b1, b2);
   ParFiniteElementSpace fes(&mesh, &fec);

   HYPRE_BigInt total_num_dofs = fes.GlobalTrueVSize();

   RT_ContinuityConstraints constraints(fes);

   cout << "Number of total DOFs:  " << total_num_dofs << '\n';
   cout << "Number of primary DOFs:" << constraints.n_primary_dofs << '\n';
   cout << "Number of entities:    " << constraints.entities.size() << '\n';

   for (int i = 0; i < constraints.entities.size(); ++i)
   {
      cout << "Entity " << i << ":\n";

      cout << "    " << "Dimension: " << constraints.entities[i].entity.dim << '\n';
      cout << "    " << "Number of DOFs: " <<
           constraints.entities[i].GetNPrimaryDofs() << '\n';
   }

   return 0;
}

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref)
{
   Mesh serial_mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   return mesh;
}
