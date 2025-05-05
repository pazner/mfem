#include "voxel_mesh.hpp"

namespace mfem
{

int FindComponents(const Table &elem_elem, Array<int> &component)
{
   const int num_elem = elem_elem.Size();
   const int *i_elem_elem = elem_elem.GetI();
   const int *j_elem_elem = elem_elem.GetJ();

   component.SetSize(num_elem);
   component = -1;

   Array<int> elem_stack(num_elem);

   int num_comp = 0;

   int stack_p = 0;
   int stack_top_p = 0; // points to the first unused element in the stack
   for (int elem = 0; elem < num_elem; elem++)
   {
      if (component[elem] >= 0) { continue; }

      component[elem] = num_comp;
      ++num_comp;

      elem_stack[stack_top_p++] = elem;

      for ( ; stack_p < stack_top_p; stack_p++)
      {
         const int i = elem_stack[stack_p];
         for (int j = i_elem_elem[i]; j < i_elem_elem[i+1]; j++)
         {
            const int k = j_elem_elem[j];
            if (component[k] < 0)
            {
               component[k] = component[i];
               elem_stack[stack_top_p++] = k;
            }
            else if (component[k] != component[i])
            {
               MFEM_ABORT("");
            }
         }
      }
   }
   return num_comp;
}

void VoxelMesh::Construct()
{
   MFEM_VERIFY(GetNE() > 0, "Empty mesh not supported.");

   h = GetElementSize(0);
   SetCurvature(0);

   Array<int> components;
   const int n_components = FindComponents(ElementToElementTable(), components);
   std::cout << "Found " << n_components << " connected components.\n";
   if (n_components > 1)
   {
      Array<int> component_count(n_components);
      component_count = 0;
      for (int c : components) { ++component_count[c]; }

      const auto max_it = std::max_element(component_count.begin(),
                                           component_count.end());
      const int c = std::distance(component_count.begin(), max_it);

      Array<Element*> new_elements(component_count[c]);
      int j = 0;
      for (int i = 0; i < GetNE(); ++i)
      {
         if (components[i] == c)
         {
            new_elements[j] = elements[i];
            ++j;
         }
         else
         {
            FreeElement(elements[i]);
         }
      }
      mfem::Swap(elements, new_elements);
      NumOfElements = j;

      // TODO: retain boundary attributes
      boundary.DeleteAll();
      NumOfBdrElements = 0;

      DeleteTables();
      RemoveUnusedVertices();
      FinalizeMesh();
   }

   Vector mmin, mmax;
   GetBoundingBox(mmin, mmax, 0);

   n.resize(Dim);
   for (int i = 0; i < Dim; ++i) { n[i] = std::ceil((mmax[i] - mmin[i])/h); }

   auto translate = [mmin](const Vector &x_old, Vector &x_new)
   {
      subtract(x_old, mmin, x_new);
   };
   VectorFunctionCoefficient translate_coeff(Dim, translate);

   Transform(translate_coeff);

   Vector center;
   std::vector<int> center_idx(Dim);
   idx2lex.resize(NumOfElements);
   for (int i = 0; i < NumOfElements; ++i)
   {
      GetElementCenter(i, center);
      for (int d = 0; d < Dim; ++d)
      {
         center_idx[d] = std::floor(center[d] / h);
      }

      LexIndex lex(center_idx.data(), Dim);
      const int lin_idx = lex.LinearIndex(n);

      idx2lex[i] = lex;
      MFEM_VERIFY(lex2idx.find(lin_idx) == lex2idx.end(), "");
      lex2idx[lin_idx] = i;
   }
}

VoxelMesh::VoxelMesh(const Mesh &mesh_) : Mesh(mesh_)
{
   Construct();
}

VoxelMesh::VoxelMesh(Mesh &&mesh_) : Mesh(std::move(mesh_))
{
   Construct();
}

VoxelMesh::VoxelMesh(const std::string &filename)
   : VoxelMesh(Mesh::LoadFromFile(filename)) { }

VoxelMesh::VoxelMesh(double h_, const std::vector<int> &n_)
   : Mesh(n_.size(), 0, 0), h(h_), n(n_) { }

VoxelMesh VoxelMesh::Coarsen() const
{
   double new_h = 2.0*h;
   std::vector<int> new_n(Dim);
   for (int i = 0; i < Dim; ++i)
   {
      new_n[i] = static_cast<int>(std::ceil(0.5*n[i]));
   }
   VoxelMesh coarsened_mesh(new_h, new_n);

   std::vector<int> new_n_vert(Dim);
   for (int i = 0; i < Dim; ++i) { new_n_vert[i] = new_n[i] + 1; }

   std::unordered_map<int,int> lex2vert;

   auto coords_to_lex = [&](const std::array<double,3> &coords)
   {
      LexIndex idx;
      idx.ndim = Dim;
      for (int d = 0; d < Dim; ++d)
      {
         idx.coords[d] = static_cast<int>(std::round(coords[d] / h));
      }
      return idx;
   };

   auto maybe_add_vertex = [&](const LexIndex &lex,
                               const std::array<double,3> &coords)
   {
      const int lin_idx = lex.LinearIndex(new_n_vert);
      if (lex2vert.find(lin_idx) == lex2vert.end())
      {
         lex2vert[lin_idx] = coarsened_mesh.NumOfVertices;
         coarsened_mesh.AddVertex(coords.data());
      }
   };

   std::array<double,3> v;

   // Add every even-indexed vertex and all of its neighbors
   for (int iv = 0; iv < NumOfVertices; ++iv)
   {
      // Populate v with the coordinates of vertex iv
      {
         const double *v_iv = vertices[iv]();
         std::copy(v_iv, v_iv + Dim, v.begin());
      }

      // Retain only the even-numbered vertices
      LexIndex lex = coords_to_lex(v);
      bool even = true;
      for (int d = 0; d < Dim; ++d)
      {
         if (lex.coords[d] % 2 != 0)
         {
            even = false;
            break;
         }
      }
      if (!even) { continue; }

      // Add the vertex to the coarse mesh (if it doesn't exist already)
      for (int d = 0; d < Dim; ++d) { lex.coords[d] /= 2; }
      maybe_add_vertex(lex, v);

      // Add all of its neighbors to the coarse mesh (if they don't exist)
      const int ngrid = std::pow(3, Dim);

      std::array<int,3> shift;
      for (int i = 0; i < ngrid; ++i)
      {
         int j = i;
         bool in_bounds = true;
         for (int d = 0; d < Dim; ++d)
         {
            shift[d] = (j % 3) - 1;
            j /= 3;

            if (lex.coords[d] + shift[d] < 0 ||
                lex.coords[d] + shift[d] >= new_n_vert[d])
            {
               in_bounds = false; break;
            }
         }

         if (!in_bounds) { continue; }

         for (int d = 0; d < Dim; ++d)
         {
            v[d] += shift[d]*new_h;
            lex.coords[d] += shift[d];
         }

         // If this vertex hasn't been added yet, add it.
         maybe_add_vertex(lex, v);

         // Reset (shift back)
         for (int d = 0; d < Dim; ++d)
         {
            v[d] -= shift[d]*new_h;
            lex.coords[d] -= shift[d];
         }
      }
   }

   // Get the vertex integration rule for the geometry
   const IntegrationRule &ir = *Geometries.GetVertices(GetElementGeometry(0));

   // Having added all the (potential) coarse mesh vertices, we now add the
   // coarse mesh elements
   for (int ie = 0; ie < NumOfElements; ++ie)
   {
      LexIndex lex = idx2lex[ie];
      // Figure out which macro element we're in
      for (int d = 0; d < Dim; ++d) { lex.coords[d] /= 2; }

      const int lin_idx = lex.LinearIndex(new_n);
      if (coarsened_mesh.lex2idx.find(lin_idx) == coarsened_mesh.lex2idx.end())
      {
         const int attr = GetAttribute(ie);
         coarsened_mesh.lex2idx[lin_idx] = coarsened_mesh.NumOfElements;
         coarsened_mesh.idx2lex.push_back(lex);

         std::vector<int> el_vert(ir.Size());
         for (int iv = 0; iv < ir.Size(); ++iv)
         {
            std::array<double,3> ip;
            ir[iv].Get(ip.data(), Dim);
            for (int d = 0; d < Dim; ++d) { lex.coords[d] += ip[d]; }
            const int v_lin_idx = lex.LinearIndex(new_n_vert);
            el_vert[iv] = lex2vert.at(v_lin_idx);
            // Reset
            for (int d = 0; d < Dim; ++d) { lex.coords[d] -= ip[d]; }
         }

         if (Dim == 1) { MFEM_ABORT("To be implemented"); }
         else if (Dim == 2) { coarsened_mesh.AddQuad(el_vert.data(), attr); }
         else if (Dim == 3) { coarsened_mesh.AddHex(el_vert.data(), attr); }
         else { MFEM_ABORT("Unsupported dimension."); }
      }
   }

   coarsened_mesh.RemoveUnusedVertices();
   coarsened_mesh.FinalizeMesh();

   const int ngrid = pow(2, Dim); // number of fine elements per coarse elements
   // Inherit boundary attributes from parents
   std::unordered_map<int,std::unordered_map<int,int>> el_to_bdr_attr;
   for (int ib = 0; ib < GetNBE(); ++ib)
   {
      const int attr = GetBdrAttribute(ib);
      int ie, info;
      GetBdrElementAdjacentElement(ib, ie, info);
      const int local_side = info/64; // decode info
      el_to_bdr_attr[ie][local_side] = attr;
   }

   for (int ib = 0; ib < coarsened_mesh.GetNBE(); ++ib)
   {
      int attr = -1;
      int ie, info;
      coarsened_mesh.GetBdrElementAdjacentElement(ib, ie, info);
      const int local_side = info/64;
      LexIndex lex = coarsened_mesh.GetLexicographicIndex(ie);
      LexIndex shifted_lex = lex;
      // Convert from coarse lexicographic index to fine index
      for (int d = 0; d < Dim; ++d)
      {
         lex.coords[d] *= 2;
      }
      for (int i = 0; i < ngrid; ++i)
      {
         int j = i;
         std::array<int,3> shift;
         for (int d = 0; d < Dim; ++d)
         {
            shift[d] = j % 2;
            j /= 2;

            shifted_lex.coords[d] = lex.coords[d] + shift[d];
         }
         const int fine_idx = GetElementIndex(shifted_lex);
         const auto result_1 = el_to_bdr_attr.find(fine_idx);

         if (result_1 != el_to_bdr_attr.end())
         {
            const auto &local_attr_map = result_1->second;
            const auto result_2 = local_attr_map.find(local_side);
            if (result_2 != local_attr_map.end())
            {
               if (attr >= 0)
               {
                  attr = std::min(attr, result_2->second);
               }
               else
               {
                  attr = result_2->second;
               }
            }
         }
      }
      MFEM_VERIFY(attr >= 0, "");
      coarsened_mesh.SetBdrAttribute(ib, attr);
   }

   return coarsened_mesh;
}

void GetVoxelParents(const VoxelMesh &coarse_mesh, const VoxelMesh &fine_mesh,
                     Array<ParentIndex> &parents, Array<int> &parent_offsets)
{
   const int dim = coarse_mesh.Dimension();
   const int coarse_ne = coarse_mesh.GetNE();
   const int ngrid = pow(2, dim); // number of fine elements per coarse elements

   parent_offsets.SetSize(coarse_ne + 1);
   parents.SetSize(ngrid*coarse_ne);

   int offset = 0;
   for (int ie = 0; ie < coarse_ne; ++ie)
   {
      parent_offsets[ie] = offset;
      LexIndex lex = coarse_mesh.GetLexicographicIndex(ie);

      // Convert from coarse lexicographic index to fine index
      for (int d = 0; d < dim; ++d)
      {
         lex.coords[d] *= 2;
      }

      for (int i = 0; i < ngrid; ++i)
      {
         int j = i;
         std::array<int,3> shift;
         for (int d = 0; d < dim; ++d)
         {
            shift[d] = j % 2;
            j /= 2;

            lex.coords[d] += shift[d];
         }

         const int fine_idx = fine_mesh.GetElementIndex(lex);
         if (fine_idx >= 0)
         {
            parents[offset] = {fine_idx, i};
            ++offset;
         }

         // Reset
         for (int d = 0; d < dim; ++d)
         {
            lex.coords[d] -= shift[d];
         }
      }
   }
   parent_offsets.Last() = offset;
   parents.SetSize(offset);
}

}
