#ifndef HDIV_PROLONGATION_HPP
#define HDIV_PROLONGATION_HPP

#include "mfem.hpp"
#include <set>

namespace mfem
{

std::vector<int> CartesianSubset(
   const std::vector<int> &sizes,
   const std::vector<std::pair<int,int>> &ranges,
   const int offset = 0)
{
   // Number of dimensions
   const int ndim = sizes.size();

   auto decode_index = [](const int i_s, const int n)
   {
      return (i_s >= 0) ? i_s : n + i_s + 1;
   };

   // Compute size of the requested subset (product of sizes in each dimension)
   std::vector<int> sub_sizes(ndim);
   for (int d = 0; d < ndim; ++d)
   {
      const int i1 = decode_index(ranges[d].first, sizes[d]);
      const int i2 = decode_index(ranges[d].second, sizes[d]);
      const int sz = i2 - i1;
      sub_sizes[d] = sz;
   }
   const int subset_size = std::accumulate(
                              sub_sizes.begin(), sub_sizes.end(), 1, std::multiplies<int>());

   std::vector<int> subset(subset_size, offset);
   for (int i = 0; i < subset_size; ++i)
   {
      // Convert from linear (subset) index into Cartesian index
      std::vector<int> idx(ndim);
      int j = i;
      for (int d = 0; d < ndim; ++d)
      {
         idx[d] = decode_index(ranges[d].first, sizes[d]) + (j % sub_sizes[d]);
         j /= sub_sizes[d];
      }
      // Convert from Cartesian index into linear (superset) index
      for (int d = 0; d < ndim; ++d)
      {
         subset[i] += idx[d] * (d == 0 ? 1 : sizes[d-1]);
      }
   }
   return subset;
}

std::vector<int> RT_CartesianSubset(
   const int p,
   const int dim,
   const std::vector<std::pair<int,int>> &ranges)
{
   std::vector<int> dofs;

   for (int d = 0; d < dim; ++d)
   {
      std::vector<int> sizes(dim, p);
      sizes[d] += 1;

      const int offset = d*pow(p, dim-1)*(p+1);
      std::vector<int> dofs_d = CartesianSubset(sizes, ranges, offset);
      dofs.insert(dofs.end(), dofs_d.begin(), dofs_d.end());
   }
   return dofs;
}

struct LocalIndexOrientation
{
   const int index, orientation;
   LocalIndexOrientation(const int i, const int o) : index(i), orientation(o) { }
};

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

   const IntegrationRule &GetIntegrationRule(const int npts) const
   {
      const int order = (dim == 0) ? 1 : 2*npts - 1;
      return IntRules.Get(GetGeometry(), order);
   }

   LocalIndexOrientation GetLocalIndexOrientation(const int element_index) const
   {
      const int mesh_dim = mesh.Dimension();
      Array<int> indices, orientations;

      if (dim == 0)
      {
         mesh.GetElementVertices(element_index, indices);
      }
      else if (dim == mesh_dim)
      {
         return LocalIndexOrientation(0, 0);
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
      const int i = std::distance(indices.begin(), found);
      const int o = (orientations.Size() > 0) ? orientations[i] : 0;
      return LocalIndexOrientation(i, o);
   }

   IntegrationPoint Transform(const IntegrationPoint &ip,
                              const LocalIndexOrientation &io) const
   {
      MFEM_VERIFY(mesh.Dimension() == 2, "Unsupported");

      if (dim == 0)
      {
         const Geometry::Type geom = mesh.GetElementGeometry(0);
         return (*Geometries.GetVertices(geom))[io.index];
      }
      else if (dim == 1)
      {
         IntegrationPoint ipt;
         ipt.Init(0);
         const double t = (io.orientation > 0) ? ip.x : 1.0 - ip.x;
         // const double x2 = (io.index == 0 || io.index == 3) ? 0.0 : 1.0;
         if (io.index == 0) { ipt.x = t; ipt.y = 0; }
         else if (io.index == 1) { ipt.x = 1; ipt.y = t; }
         else if (io.index == 2) { ipt.x = 1 - t; ipt.y = 1; }
         else if (io.index == 3) { ipt.x = 0; ipt.y = 1 - t; }
         else { MFEM_ABORT(""); }
         return ipt;
      }
      else // dim == 2
      {
         // No need to transform element interior DOFs
         return ip;
      }
   }
};

struct EntityDofs; // Forward declaration

struct ElementEntityDofs
{
   const Entity &entity;
   const int element_index;
   LocalIndexOrientation local_index_orientation;
   Array<int> local_dofs;
   Array<int> global_dofs;
   IntegrationRule ir_el;

   ElementEntityDofs(const FiniteElementSpace &fes, const Entity &entity_,
                     const int element_index_)
      : entity(entity_),
        element_index(element_index_),
        local_index_orientation(entity.GetLocalIndexOrientation(element_index))
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
      const int loc = local_index_orientation.index;

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

         local_dofs[0] = x_offset*p + y_offset*(p+1)*(p-1);
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
            local_dofs[ix] = offset_x + (ix+1)*stride_x;
         }

         const int ndof_per_dim = p*(p+1);

         const int offset_y = (loc == 1) ? p - 1 : (loc == 2) ? p*p : 0;
         for (int iy = 0; iy < ny; ++iy)
         {
            // (iy + 1) below since skipping the first DOF (vertex)
            local_dofs[nx + iy] = ndof_per_dim + offset_y + (iy+1)*stride_y;
         }

         // std::vector<int> sizes_x = {p+1, p};
         // std::vector<int> sizes_y = {p, p+1};
         std::vector<std::pair<int,int>> ranges;

         std::pair<int,int> L = {0, 1};
         std::pair<int,int> R = {-2, -1};
         std::pair<int,int> I = {1, -2};

         if (loc == 0) { ranges = {I, L}; }
         else if (loc == 1) { ranges = {R, I}; }
         else if (loc == 2) { ranges = {I, R}; }
         else if (loc == 3) { ranges = {L, I}; }

         auto dofs = RT_CartesianSubset(p, dim, ranges);

         // auto dofs_x = CartesianSubset(sizes_x, ranges);
         // auto dofs_y = CartesianSubset(sizes_y, ranges, p*(p+1));
         // dofs_x.insert(dofs_x.end(), dofs_y.begin(), dofs_y.end());

         // std::cout << "loc = " << loc << '\n';
         // for (int i = 0; i < dofs.size(); ++i)
         // {
         //    std::cout << dofs[i];
         //    if (i < dofs.size() - 1) { std::cout << ", "; }
         //    else { std::cout << '\n'; }
         // }
         // for (int i = 0; i < local_dofs.Size(); ++i)
         // {
         //    std::cout << local_dofs[i];
         //    if (i < local_dofs.Size() - 1) { std::cout << ", "; }
         //    else { std::cout << "\n\n"; }
         // }
      }
      else if (entity.dim == 2)
      {
         // Exclude edges --- only include interior element DOFs
         local_dofs.SetSize(2*(p-1)*(p-2));
         local_dofs = -1.0;

         int idx = 0;

         // x-parallel vectors
         // Skip first and last
         for (int iy = 1; iy < p - 1; ++iy)
         {
            for (int ix = 1; ix < p; ++ix)
            {
               local_dofs[idx] = ix + iy*(p+1);
               ++idx;
            }
         }

         const int offset = p*(p+1);
         // y-parallel vectors
         // Skip first and last
         for (int iy = 1; iy < p; ++iy)
         {
            for (int ix = 1; ix < p - 1; ++ix)
            {
               local_dofs[idx] = offset + ix + iy*p;
               ++idx;
            }
         }

         std::vector<int> sizes_x = {p+1, p};
         std::vector<int> sizes_y = {p, p+1};
         std::vector<std::pair<int,int>> ranges_x = {{1, -2}, {1, -2}};
         std::vector<std::pair<int,int>> ranges_y = {{1, -2}, {1, -2}};
         auto dofs_x = CartesianSubset(sizes_x, ranges_x);
         auto dofs_y = CartesianSubset(sizes_y, ranges_y, p*(p+1));

         dofs_x.insert(dofs_x.end(), dofs_y.begin(), dofs_y.end());

         // std::cout << "element: \n";
         // for (int i = 0; i < dofs_x.size(); ++i)
         // {
         //    std::cout << dofs_x[i];
         //    if (i < dofs_x.size() - 1) { std::cout << ", "; }
         //    else { std::cout << '\n'; }
         // }
         // for (int i = 0; i < local_dofs.Size(); ++i)
         // {
         //    std::cout << local_dofs[i];
         //    if (i < local_dofs.Size() - 1) { std::cout << ", "; }
         //    else { std::cout << "\n\n"; }
         // }
      }

      auto set_global_dofs = [&](Array<int> &local, Array<int> &global)
      {
         global.SetSize(local.Size());
         for (int i = 0; i < local.Size(); ++i)
         {
            const int sd = dof_map[local[i]];
            const int d = (sd >= 0) ? sd : -1 - sd;
            // Remap from lexicographic to MFEM-native ordering
            local[i] = d;
            // global will incorporate the sign (orientation) of the DOF
            global[i] = element_dofs[d];
         }
      };

      set_global_dofs(local_dofs, global_dofs);

      // Set up the integration rules
      const int ndof = local_dofs.Size();
      const int npts = std::ceil(0.5*ndof);
      const IntegrationRule &ir = entity.GetIntegrationRule(npts);
      ir_el.SetSize(ir.Size());

      for (int i = 0; i < ir.Size(); ++i)
      {
         ir_el[i] = entity.Transform(ir[i], local_index_orientation);
      }
   }

   DenseMatrix FormDofMatrix(const FiniteElementSpace &fes,
                             const IntegrationRule &ir) const
   {
      const int ndof = local_dofs.Size();
      const int npts = ir.Size();

      const FiniteElement &fe = *fes.GetFE(element_index);
      const int sdim = fe.GetDim();
      ElementTransformation &T = *fes.GetElementTransformation(element_index);
      DenseMatrix shape(fe.GetDof(), sdim);
      DenseMatrix V(npts*sdim, ndof);
      for (int i = 0; i < npts; ++i)
      {
         const IntegrationPoint &ip = ir[i];
         T.SetIntPoint(&ip);
         fe.CalcPhysVShape(T, shape);
         for (int j = 0; j < ndof; ++j)
         {
            for (int d = 0; d < sdim; ++d)
            {
               V(d + i*sdim, j) = shape(local_dofs[j], d);
            }
         }
      }
      return V;
   }

   DenseMatrix FormDofMatrix(const FiniteElementSpace &fes) const
   {
      return FormDofMatrix(fes, ir_el);
   }
};

struct ConstrainedElementEntity
{
   const Entity &entity;
   ElementEntityDofs dofs;
   const ConstrainedElementEntity &primary;
   const int dof_offset;
   bool is_primary;

   std::vector<const ConstrainedElementEntity*> children;
   DenseMatrix Pi;
   std::vector<DenseMatrix> Pi_sub;

   ConstrainedElementEntity(const FiniteElementSpace &fes,
                            const Entity &entity_,
                            int element_index,
                            int dof_offset_)
      : entity(entity_),
        dofs(fes, entity, element_index),
        primary(*this),
        dof_offset(dof_offset_),
        is_primary(true)
   {

   }

   ConstrainedElementEntity(const FiniteElementSpace &fes,
                            const Entity &entity_,
                            int element_index,
                            const std::vector<EntityDofs> &entities,
                            int dof_offset_)
      : ConstrainedElementEntity(fes, entity_, *this, element_index, entities,
                                 dof_offset_)
   {
      is_primary = true;
   }

   ConstrainedElementEntity(const FiniteElementSpace &fes,
                            const Entity &entity_,
                            const ConstrainedElementEntity &primary_,
                            int element_index,
                            const std::vector<EntityDofs> &entities,
                            int dof_offset_)
      : entity(entity_),
        dofs(fes, entity, element_index),
        primary(primary_),
        dof_offset(dof_offset_)
   {
      is_primary = false;

      // This is an edge: add the vertices as children
      if (entity.dim == 1)
      {
         Array<int> v;
         fes.GetMesh()->GetEdgeVertices(entity.entity_index, v);
         for (const int iv : v)
         {
            AddChild(entities[iv]);
         }
      }

      const IntegrationRule &ir_1 = dofs.ir_el;
      const IntegrationRule &ir_2 = primary.dofs.ir_el;

      // Evaluate the constrained DOFs at the constraining points
      DenseMatrix Phi = dofs.FormDofMatrix(fes, ir_1);
      const int ndof1 = Phi.Width();
      Phi.Transpose();
      DenseMatrix PhiTPhi(ndof1, ndof1);
      MultAAt(Phi, PhiTPhi);
      DenseMatrixInverse PhiTPhi_inv(PhiTPhi);

      auto compute_constraint_matrix = [&](const DenseMatrix &Psi)
      {
         const int ndof2 = Psi.Width();
         DenseMatrix PhiTPsi(ndof1, ndof2);
         Mult(Phi, Psi, PhiTPsi);
         DenseMatrix C(ndof1, ndof2);
         PhiTPhi_inv.Mult(PhiTPsi, C);
         return C;
      };

      Pi = compute_constraint_matrix(primary.dofs.FormDofMatrix(fes, ir_2));

      for (int i = 0; i < children.size(); ++i)
      {
         const ConstrainedElementEntity &child_1 = *children[i];
         const ConstrainedElementEntity &child_2 = *primary.children[i];

         const DenseMatrix &C1 = child_1.Pi;
         const DenseMatrix &C2 = child_2.Pi;

         const DenseMatrix Psi1 = child_1.dofs.FormDofMatrix(fes, ir_1);
         const DenseMatrix Psi2 = child_2.dofs.FormDofMatrix(fes, ir_2);

         DenseMatrix Psi(Psi2.Height(), C2.Width());
         Mult(Psi2, C2, Psi);
         AddMult_a(-1.0, Psi1, C1, Psi);

         Pi_sub.emplace_back(compute_constraint_matrix(Psi));
      }
   }

   void AddChild(const EntityDofs &child_entity);

   int GetNChildrenDofs() const
   {
      int ndof_children = 0;
      for (const ConstrainedElementEntity *child : children)
      {
         ndof_children += child->dofs.local_dofs.Size();
      }
      return ndof_children;
   }
};

struct EntityDofs
{
   Entity entity;
   // First entry of constrained is the primary
   std::vector<ConstrainedElementEntity> constrained;

   EntityDofs(const FiniteElementSpace &fes, int entity_index, int entity_dim,
              const Table &el_table, int dof_offset, const std::vector<EntityDofs> &entities)
      : entity(*fes.GetMesh(), entity_dim, entity_index)
   {
      Array<int> row;
      el_table.GetRow(entity_index, row);
      constrained.reserve(row.Size());
      for (int i = 0; i < row.Size(); ++i)
      {
         if (i == 0)
         {
            constrained.emplace_back(fes, entity, row[i], entities, dof_offset);
         }
         else
         {
            const ConstrainedElementEntity &primary = constrained.front();
            constrained.emplace_back(fes, entity, primary, row[i], entities, dof_offset);
         }
      }
   }

   // Version of the constructor for elements
   EntityDofs(const FiniteElementSpace &fes, int element_index, int dof_offset,
              const std::vector<EntityDofs> &entities)
      : entity(*fes.GetMesh(), fes.GetMesh()->Dimension(), element_index)
   {
      constrained.emplace_back(fes, entity, element_index, entities, dof_offset);
   }

   int GetNPrimaryDofs() const
   {
      return constrained.front().dofs.local_dofs.Size();
   }

   std::set<int> GetGlobalDofSet() const
   {
      std::set<int> global_dof_set;
      for (const ConstrainedElementEntity &c : constrained)
      {
         const Array<int> &dofs = c.dofs.global_dofs;
         for (const int d_s : dofs)
         {
            const int d_i = (d_s >= 0) ? d_s : -1 - d_s;
            global_dof_set.insert(d_i);
         }
         // global_dof_set.insert(dofs.begin(), dofs.end());
      }
      return global_dof_set;
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
   Array<int> dof2entity;
   Array<int> bdr_dofs;
   int n_primary_dofs;
   FiniteElementSpace &fes;
   mutable SparseMatrix P;
   mutable SparseMatrix R;
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
         entities.emplace_back(fes, iv, 0, *el_table, n_primary_dofs, entities);
         n_primary_dofs += entities.back().GetNPrimaryDofs();
      }

      el_table.reset(Transpose(mesh.ElementToEdgeTable()));
      for (int ie = 0; ie < mesh.GetNEdges(); ++ie)
      {
         entities.emplace_back(fes, ie, 1, *el_table, n_primary_dofs, entities);
         n_primary_dofs += entities.back().GetNPrimaryDofs();
      }

      dof2entity.SetSize(n_primary_dofs);
      for (int i = 0; i < entities.size(); ++i)
      {
         const ConstrainedElementEntity &p = entities[i].constrained[0];
         for (int j = 0; j < p.dofs.local_dofs.Size(); ++j)
         {
            dof2entity[p.dof_offset + j] = i;
         }
      }

      for (int iel = 0; iel < mesh.GetNE(); ++iel)
      {
         // entities.emplace_back(fes, iel, n_primary_dofs);
         entities.emplace_back(fes, iel, n_primary_dofs, entities);
         n_primary_dofs += entities.back().GetNPrimaryDofs();
      }

      std::set<int> bdr_dof_vec;
      auto add_bdr_dofs = [&](const ConstrainedElementEntity &p)
      {
         const int ndof = p.dofs.local_dofs.Size();
         for (int i = 0; i < ndof; ++i)
         {
            bdr_dof_vec.emplace(p.dof_offset + i);
         }
      };

      for (int ibe = 0; ibe < mesh.GetNBE(); ++ibe)
      {
         Array<int> bv, be, orientations;
         mesh.GetBdrElementVertices(ibe, bv);
         for (const int iv : bv)
         {
            add_bdr_dofs(entities[iv].constrained[0]);
         }

         const int offset = mesh.GetNV();
         mesh.GetBdrElementEdges(ibe, be, orientations);
         for (const int ie : be)
         {
            add_bdr_dofs(entities[offset + ie].constrained[0]);
         }
      }

      bdr_dofs.SetSize(bdr_dof_vec.size());
      std::copy(bdr_dof_vec.begin(), bdr_dof_vec.end(), bdr_dofs.begin());
   }

   void FormProlongationMatrix() const
   {
      SparseMatrix P_(fes.GetTrueVSize(), n_primary_dofs);

      for (const EntityDofs &entity : entities)
      {
         // Place the interpolation matrices for constrained DOFs
         for (const ConstrainedElementEntity &constrained : entity.constrained)
         {
            auto set_constraint_submatrix = [&](const DenseMatrix &Pi,
                                                const ConstrainedElementEntity &p)
            {
               const int ndof1 = Pi.Height();
               const int ndof2 = Pi.Width();

               MFEM_VERIFY(ndof1 == constrained.dofs.local_dofs.Size(), "");
               MFEM_VERIFY(ndof2 == p.dofs.local_dofs.Size(), "");

               for (int kj = 0; kj < ndof2; ++kj)
               {
                  const int j_s = p.dofs.global_dofs[kj];
                  const int s1 = (j_s >= 0) ? 1 : -1;
                  const int j = p.dof_offset + kj;
                  for (int ki = 0; ki < ndof1; ++ki)
                  {
                     const int i_s = constrained.dofs.global_dofs[ki];
                     const int s2 = (i_s >= 0) ? 1 : -1;
                     const int i = (i_s >= 0) ? i_s : -1 - i_s;

                     const double val = Pi(ki, kj)*s1*s2;

                     const double val_prev = P_.SearchRow(i, j);
                     MFEM_ASSERT(val_prev == 0.0 || std::abs(val - val_prev) < 1e-10, "");

                     // std::cout << "Setting (" << i << "," << j << ") = " << val << '\n';
                     P_.Set(i, j, val);
                  }
               }
            };

            set_constraint_submatrix(constrained.Pi, constrained.primary);

            for (int i = 0; i < constrained.children.size(); ++i)
            {
               const ConstrainedElementEntity &c = *constrained.children[i];
               set_constraint_submatrix(constrained.Pi_sub[i], c.primary);
            }
         }
      }

      P_.Finalize();
      P.Swap(P_);
   }

   void FormRestrictionMatrix() const
   {
      SparseMatrix R_(n_primary_dofs, fes.GetTrueVSize());

      for (const EntityDofs &entity : entities)
      {
         const ConstrainedElementEntity &primary = entity.constrained[0];

         const int ndof = primary.dofs.local_dofs.Size();
         const int offset = primary.dof_offset;

         for (int idx = 0; idx < ndof; ++idx)
         {
            const int i = offset + idx;
            const int j_s = primary.dofs.global_dofs[idx];
            const int j = (j_s >= 0) ? j_s : -1 - j_s;

            R_.Set(i, j, 1.0);
         }
      }

      R_.Finalize();
      R.Swap(R_);
   }

   SparseMatrix &GetProlongationMatrix() const
   {
      if (P.Empty()) { FormProlongationMatrix(); }
      return P;
   }

   SparseMatrix &GetRestrictionMatrix() const
   {
      if (R.Empty()) { FormRestrictionMatrix(); }
      return R;
   }
};

void ConstrainedElementEntity::AddChild(const EntityDofs &child_entity)
{
   const int iel = dofs.element_index;
   for (const ConstrainedElementEntity &constrained : child_entity.constrained)
   {
      if (constrained.dofs.element_index == iel)
      {
         children.push_back(&constrained);
         return;
      }
   }
}

} // namespace mfem

#endif
