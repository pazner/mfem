#include "voxel_integ.hpp"

namespace mfem
{

template<typename TA, typename TX, typename TY>
MFEM_HOST_DEVICE inline
void AddMult(const int h, const int w, const TA *data, const TX *x, TY *y)
{
   const TA *d_col = data;
   for (int col = 0; col < w; col++)
   {
      TX x_col = x[col];
      for (int row = 0; row < h; row++)
      {
         y[row] += x_col*d_col[row];
      }
      d_col += h;
   }
}

VoxelIntegrator::VoxelIntegrator(BilinearFormIntegrator *integ_)
   : integ(integ_) { }

void VoxelIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   ne = fes.GetNE();
   if (ne == 0) { return; }

   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation &Tr = *fes.GetElementTransformation(0);

   const int vdim = fes.GetVDim();
   ndof_per_el = el.GetDof() * vdim;

   DenseMatrix elmat_p;
   integ->AssembleElementMatrix(el, Tr, elmat_p);

   // Reorder from "native" ordering to lexicographic
   if (auto *nfe = dynamic_cast<const NodalFiniteElement*>(&el))
   {
      const Array<int> lex = nfe->GetLexicographicOrdering();
      const int n = lex.Size();
      elmat.SetSize(elmat_p.Height(), elmat_p.Width());
      for (int vd1 = 0; vd1 < vdim; ++vd1)
      {
         const int o1 = vd1*n;
         for (int vd2 = 0; vd2 < vdim; ++ vd2)
         {
            const int o2 = vd2*n;
            for (int j = 0; j < n; ++j)
            {
               for (int i = 0; i < n; ++i)
               {
                  elmat(i + o1, j + o2) = elmat_p(lex[i] + o1, lex[j] + o2);
               }
            }
         }
      }
   }
   else
   {
      elmat = elmat_p;
   }
}

void VoxelIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   const int h = elmat.Height();
   const int w = elmat.Width();
   const double *d_elmat = elmat.GetData();
   const double *d_x = x.GetData();
   double *d_y = y.GetData();

   for (int e = 0; e < ne; ++e)
   {
      AddMult(h, w, d_elmat, d_x, d_y);
      d_x += ndof_per_el;
      d_y += ndof_per_el;
   }
}

void VoxelIntegrator::AssembleDiagonalPA(Vector &diag)
{
   for (int e = 0; e < ne; ++e)
   {
      for (int i = 0; i < ndof_per_el; ++i)
      {
         diag[i + e*ndof_per_el] = elmat(i,i);
      }
   }
}

}
