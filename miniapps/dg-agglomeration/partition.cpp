#include "partition.hpp"

#ifdef MFEM_USE_METIS_5
#include "metis.h"
#else
#error "METIS is required"
#endif

#include <algorithm>
#include <functional>
#include <vector>

namespace mfem
{

namespace
{

/// True if the graph (I, J) with n vertices is connected (or empty).
bool IsGraphConnected(int n, const Array<idx_t> &I, const Array<idx_t> &J)
{
   if (n == 0) { return true; }
   std::vector<char> visited(n, 0);
   std::vector<int> stack;
   stack.push_back(0);
   visited[0] = 1;
   int count = 1;
   while (!stack.empty())
   {
      const int v = stack.back();
      stack.pop_back();
      for (idx_t k = I[v]; k < I[v+1]; k++)
      {
         const int u = static_cast<int>(J[k]);
         if (!visited[u])
         {
            visited[u] = 1;
            count++;
            stack.push_back(u);
         }
      }
   }
   return count == n;
}

} // anonymous namespace

Array<int> PartitionMesh(Mesh &mesh, const int npart, const int part_method,
                         OptRef<Array<int>> subset)
{
   // On a ParMesh this is the number of rank-local elements.
   const int ne_mesh = mesh.GetNE();
   const int ne = subset ? subset->Size() : ne_mesh;

   Array<int> partitioning(ne);

   // Early return special cases for one partition requested, or more
   // partitions than elements.
   if (npart <= 1)
   {
      partitioning = 0;
      return partitioning;
   }
   else if (ne <= npart)
   {
      for (int i = 0; i < ne; i++) { partitioning[i] = i; }
      return partitioning;
   }

   // Build the (sub)graph in CSR format.
   //
   // graph_id[e] = vertex index of mesh element e in the graph, or -1 if e is
   // not part of it. Neighbors are kept only if they are rank-local elements
   // in the graph. This matters on a ParMesh: once face-neighbor data has been
   // exchanged, ElementToElementTable() may contain face-neighbor elements
   // (owned by other ranks) with indices >= GetNE(), which must not be given
   // to METIS.
   std::vector<int> graph_id(ne_mesh, -1);
   for (int r = 0; r < ne; r++)
   {
      const int el = subset ? (*subset)[r] : r;
      MFEM_ASSERT(el >= 0 && el < ne_mesh, "Invalid element index in subset.");
      graph_id[el] = r;
   }

   const Table &e2e = mesh.ElementToElementTable();
   const int *iI = e2e.HostReadI();
   const int *iJ = e2e.HostReadJ();

   Array<idx_t> I(ne + 1), J;
   I[0] = 0;
   for (int r = 0; r < ne; r++)
   {
      const int el = subset ? (*subset)[r] : r;
      for (int k = iI[el]; k < iI[el+1]; k++)
      {
         const int nb = iJ[k];
         if (nb < 0 || nb >= ne_mesh || nb == el) { continue; }
         const int v = graph_id[nb];
         if (v >= 0) { J.Append(static_cast<idx_t>(v)); }
      }
      I[r+1] = J.Size();
   }

   idx_t options[METIS_NOPTIONS];
   METIS_SetDefaultOptions(options);
   // Request contiguous parts only if the graph is connected; METIS errors out
   // otherwise. Rank-local subdomains and sub-aggregates are not guaranteed to
   // be connected.
   options[METIS_OPTION_CONTIG] = IsGraphConnected(ne, I, J) ? 1 : 0;

   // Sort the neighbor lists
   if (part_method >= 0 && part_method <= 2)
   {
      for (int i = 0; i < ne; i++)
      {
         // Sort in decreasing order, as in previous versions of MFEM.
         std::sort(J.GetData() + I[i], J.GetData() + I[i+1],
                   std::greater<idx_t>());
      }
   }

   idx_t n = ne;
   idx_t ncon = 1;
   idx_t mpart = npart;
   idx_t edgecut;
   Array<idx_t> p(ne);

   if (part_method == 0 || part_method == 3)
   {
      // Should be used to partition a graph into a small number of partitions
      // (less than 8).
      const int err = METIS_PartGraphRecursive(
                         &n, &ncon, I.GetData(), J.GetData(), NULL, NULL, NULL,
                         &mpart, NULL, NULL, options, &edgecut, p.GetData());
      MFEM_VERIFY(err == METIS_OK, "Error in METIS_PartGraphRecursive");
   }
   else if (part_method == 1 || part_method == 4)
   {
      // Should be used to partition a graph into a large number of partitions
      // (greater than 8).
      const int err = METIS_PartGraphKway(
                         &n, &ncon, I.GetData(), J.GetData(), NULL, NULL, NULL,
                         &mpart, NULL, NULL, options, &edgecut, p.GetData());
      MFEM_VERIFY(err == METIS_OK, "Error in METIS_PartGraphKway");
   }
   else
   {
      MFEM_ABORT("Unsupported part_method: " << part_method);
   }

   for (int i = 0; i < ne; i++) { partitioning[i] = static_cast<int>(p[i]); }

   // Note: METIS can occasionally return empty parts. Agglomerate() tolerates
   // this (an empty part becomes an empty aggregate that contributes no coarse
   // dofs), so no fix-up is applied here.
   return partitioning;
}

} // namespace mfem
