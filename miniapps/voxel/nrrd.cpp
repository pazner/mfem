#include "mfem.hpp"

using namespace std;
using namespace mfem;

struct LexIndex
{
   using Idx = array<int, 3>;

   int ndim = 0;
   Idx dims = {};
   int size;

   LexIndex(const vector<int> &dims_) : ndim(dims_.size())
   {
      copy(dims_.begin(), dims_.end(), dims.begin());
      size = 1;
      for (int d = 0; d < ndim; ++d) { size *= dims[d]; }
   }

   Idx CartesianIndex(int i) const
   {
      Idx cart;
      for (int d = 0; d < ndim; ++d)
      {
         cart[d] = i % dims[d];
         i /= dims[d];
      }
      return cart;
   }

   int LinearIndex(const Idx &i) const
   {
      int stride = 1;
      int idx = 0;
      for (int d = 0; d < ndim; ++d)
      {
         idx += i[d] * stride;
         stride *= dims[d];
      }
      return idx;
   }

   int Size() const { return size; }
};

Mesh ReadNrrd(const string &header_file)
{
   map<string,string> header;

   ifstream f(header_file);
   string line;
   getline(f, line); // Skip first line
   while (getline(f, line))
   {
      if (line.front() == '#') { continue; }
      auto pos = line.find(':');
      if (pos != string::npos)
      {
         const string field_name = line.substr(0, pos);
         string value = line.substr(pos + 2, string::npos);
         if (value.back() == '\r')
         {
            value.resize(value.size() - 1);
         }
         header[field_name] = value;
      }
   }

   const string data_file = header["data file"];
   const int dim = stoi(header["dimension"]);
   const string sizes_str = header["sizes"];
   string directions = header["space directions"];

   vector<int> cell_dims(dim), vert_dims(dim);
   {
      istringstream s(sizes_str);
      for (int d = 0; d < dim; ++d)
      {
         s >> cell_dims[d];
         vert_dims[d] = cell_dims[d] + 1;
      }
   }

   Vector phys_dims(dim);
   {
      for (char &c : directions)
      {
         if (c == '(' || c == ',' || c == ')')
         {
            c = ' ';
         }
      }
      istringstream s(directions);

      for (int d = 0; d < dim; ++d)
      {
         double val;
         for (int d2 = 0; d2 < d; ++d2) { s >> val; }
         s >> phys_dims[d];
         for (int d2 = d + 1; d2 < dim; ++d2) { s >> val; }
      }
   }

   int total_cells = 1;
   for (int d = 0; d < dim; ++d) { total_cells *= cell_dims[d]; }
   int total_verts = 1;
   for (int d = 0; d < dim; ++d) { total_verts *= vert_dims[d]; }

   vector<uint8_t> cells(total_cells);
   {
      ifstream f(header["data file"], ios::binary);
      f.read(reinterpret_cast<char*>(cells.data()), total_cells);
   }

   LexIndex cell_idx(cell_dims);
   LexIndex vert_idx(vert_dims);
   LexIndex offset_idx(vector<int>(dim, 2));

   vector<int> lex2cell(total_cells, -1);
   vector<int> lex2vert(total_verts, -1);
   vector<int> cell2lex;
   vector<int> vert2lex;

   cell2lex.reserve(total_cells);
   vert2lex.reserve(total_verts);

   int cell_i = 0;
   int vert_i = 0;
   for (int i = 0; i < total_cells; ++i)
   {
      if (cells[i] != 0)
      {
         lex2cell[i] = cell_i;
         cell2lex.push_back(i);
         ++cell_i;

         auto cart = cell_idx.CartesianIndex(i);
         for (int o = 0; o < offset_idx.Size(); ++o)
         {
            auto cart_offset = cart;
            const auto offset = offset_idx.CartesianIndex(o);
            for (int d = 0; d < dim; ++d) { cart_offset[d] += offset[d]; }
            const int iv_lex = vert_idx.LinearIndex(cart_offset);
            if (lex2vert[iv_lex] < 0)
            {
               lex2vert[iv_lex] = vert_i;
               vert2lex.push_back(iv_lex);
               ++vert_i;
            }
         }
      }
   }

   Mesh mesh(dim, vert_i, cell_i);

   for (int i = 0; i < vert_i; ++i)
   {
      const int lex = vert2lex[i];
      const auto cart = vert_idx.CartesianIndex(lex);
      array<double,3> coords;
      for (int d = 0; d < dim; ++d)
      {
         coords[d] = cart[d]*phys_dims[d] / double(cell_dims[d]);
      }
      mesh.AddVertex(coords.data());
   }

   for (int i = 0; i < cell_i; ++i)
   {
      if (dim == 2)
      {
         MFEM_ABORT("");
      }
      else if (dim == 3)
      {
         array<int,8> verts;
         const int cell_lex = cell2lex[i];
         const auto cell_cart = cell_idx.CartesianIndex(cell_lex);
         const int vert_lex = vert_idx.LinearIndex(cell_cart);
         verts[0] = lex2vert[vert_lex];
         verts[1] = lex2vert[vert_lex + 1];
         verts[2] = lex2vert[vert_lex + 1 + vert_dims[0]];
         verts[3] = lex2vert[vert_lex + vert_dims[0]];

         const int off = vert_dims[0] * vert_dims[1];

         verts[4] = lex2vert[vert_lex + off];
         verts[5] = lex2vert[vert_lex + off + 1];
         verts[6] = lex2vert[vert_lex + off + 1 + vert_dims[0]];
         verts[7] = lex2vert[vert_lex + off + vert_dims[0]];

         mesh.AddHex(verts.data());
      }
      else
      {
         MFEM_ABORT("");
      }
   }

   mesh.FinalizeMesh();

   return mesh;
}

int main(int argc, char *argv[])
{
   string header_file = "Berea.nhdr";

   OptionsParser args(argc, argv);
   args.AddOption(&header_file, "-m", "--mesh", "NRRD header file to use.");
   args.ParseCheck();

   Mesh mesh = ReadNrrd(header_file);

   mesh.Save("mesh.mesh");

   return 0;
}
