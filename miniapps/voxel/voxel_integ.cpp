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

VoxelBlockJacobi::VoxelBlockJacobi(
   ParFiniteElementSpace &fes,
   VoxelIntegrator &integ,
   const Array<int> &ess_dofs_,
   double damping)
   : Solver(fes.GetTrueVSize()),
     ne(fes.GetNE()),
     vdim(fes.GetVDim()),
     ntdof(fes.GetTrueVSize() / vdim),
     ndof_per_el(ne > 0 ? fes.GetFE(0)->GetDof() : 0),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ess_dofs(ess_dofs_)
{
   const int vdim2 = vdim*vdim;
   ParFiniteElementSpace matrix_fes(fes.GetParMesh(), fes.FEColl(), vdim2,
                                    Ordering::byVDIM);
   integ.AssemblePA(fes);

   const DenseMatrix &elmat = integ.GetElementMatrix();
   Vector blockdiag_evec(vdim2*ndof_per_el*fes.GetNE());
   auto d_blockdiag_evec = Reshape(blockdiag_evec.HostWrite(), ndof_per_el, vdim,
                                   vdim, ne);

   for (int e = 0; e < fes.GetNE(); ++e)
   {
      for (int vd1 = 0; vd1 < vdim; ++vd1)
      {
         for (int vd2 = 0; vd2 < vdim; ++vd2)
         {
            for (int i = 0; i < ndof_per_el; ++i)
            {
               d_blockdiag_evec(i, vd1, vd2, e) =
                  elmat(i + vd1*ndof_per_el, i + vd2*ndof_per_el);
            }
         }
      }
   }

   const Operator &elem_restr = *matrix_fes.GetElementRestriction(
                                   ElementDofOrdering::LEXICOGRAPHIC);
   Vector blockdiag_lvec(elem_restr.Width());
   elem_restr.MultTranspose(blockdiag_evec, blockdiag_lvec);
   blockdiag_evec.Destroy(); // free memory

   const Operator *P = matrix_fes.GetProlongationMatrix();
   MFEM_VERIFY(P != nullptr, "");

   blockdiag_tvec.SetSize(P->Width());
   P->MultTranspose(blockdiag_lvec, blockdiag_tvec);

   for (int i = 0; i < ntdof; ++i)
   {
      DenseMatrix mat(&blockdiag_tvec(i*vdim2), vdim, vdim);
      mat.Invert();
      mat *= damping;
   }
}

void VoxelBlockJacobi::Mult(const Vector &x, Vector &y) const
{
   const int vd = vdim;
   const bool t = byvdim;
   const auto d_x = Reshape(x.HostRead(), t?vd:ntdof, t?ntdof:vd);
   const auto d_blockdiag = Reshape(blockdiag_tvec.HostRead(), vd, vd, ntdof);
   auto d_y = Reshape(y.HostWrite(), t?vd:ntdof, t?ntdof:vd);

   for (int i = 0; i < ntdof; ++i)
   {
      for (int vd1 = 0; vd1 < vdim; ++vd1)
      {
         double val = 0.0;
         for (int vd2 = 0; vd2 < vdim; ++vd2)
         {
            val += d_blockdiag(vd1, vd2, i)*d_x(t?vd2:i, t?i:vd2);
         }
         d_y(t?vd1:i, t?i:vd1) = val;
      }
   }

   for (int i : ess_dofs) { y[i] = 0.0; }
}

VoxelChebyshev::VoxelChebyshev(
   const Operator &op_,
   ParFiniteElementSpace &fes,
   VoxelIntegrator &integ,
   const Array<int> &ess_dofs_,
   const int order_)
   : Solver(fes.GetTrueVSize()),
     op(op_),
     order(order_),
     block_jacobi(fes, integ, ess_dofs_),
     coeffs(order)
{
   ProductOperator po(&block_jacobi, &op, false, false);
   PowerMethod pm(fes.GetComm());

   Vector eigvec(op.Width());
   max_eig_estimate = pm.EstimateLargestEigenvalue(po, eigvec);

   // Set up Chebyshev coefficients
   // For reference, see e.g., Parallel multigrid smoothing: polynomial versus
   // Gauss-Seidel by Adams et al.
   const double upper_bound = 1.2 * max_eig_estimate;
   const double lower_bound = 0.3 * max_eig_estimate;
   const double theta = 0.5 * (upper_bound + lower_bound);
   const double delta = 0.5 * (upper_bound - lower_bound);

   switch (order-1)
   {
      case 0:
      {
         coeffs[0] = 1.0 / theta;
         break;
      }
      case 1:
      {
         double tmp_0 = 1.0/(pow(delta, 2) - 2*pow(theta, 2));
         coeffs[0] = -4*theta*tmp_0;
         coeffs[1] = 2*tmp_0;
         break;
      }
      case 2:
      {
         double tmp_0 = 3*pow(delta, 2);
         double tmp_1 = pow(theta, 2);
         double tmp_2 = 1.0/(-4*pow(theta, 3) + theta*tmp_0);
         coeffs[0] = tmp_2*(tmp_0 - 12*tmp_1);
         coeffs[1] = 12/(tmp_0 - 4*tmp_1);
         coeffs[2] = -4*tmp_2;
         break;
      }
      case 3:
      {
         double tmp_0 = pow(delta, 2);
         double tmp_1 = pow(theta, 2);
         double tmp_2 = 8*tmp_0;
         double tmp_3 = 1.0/(pow(delta, 4) + 8*pow(theta, 4) - tmp_1*tmp_2);
         coeffs[0] = tmp_3*(32*pow(theta, 3) - 16*theta*tmp_0);
         coeffs[1] = tmp_3*(-48*tmp_1 + tmp_2);
         coeffs[2] = 32*theta*tmp_3;
         coeffs[3] = -8*tmp_3;
         break;
      }
      case 4:
      {
         double tmp_0 = 5*pow(delta, 4);
         double tmp_1 = pow(theta, 4);
         double tmp_2 = pow(theta, 2);
         double tmp_3 = pow(delta, 2);
         double tmp_4 = 60*tmp_3;
         double tmp_5 = 20*tmp_3;
         double tmp_6 = 1.0/(16*pow(theta, 5) - pow(theta, 3)*tmp_5 + theta*tmp_0);
         double tmp_7 = 160*tmp_2;
         double tmp_8 = 1.0/(tmp_0 + 16*tmp_1 - tmp_2*tmp_5);
         coeffs[0] = tmp_6*(tmp_0 + 80*tmp_1 - tmp_2*tmp_4);
         coeffs[1] = tmp_8*(tmp_4 - tmp_7);
         coeffs[2] = tmp_6*(-tmp_5 + tmp_7);
         coeffs[3] = -80*tmp_8;
         coeffs[4] = 16*tmp_6;
         break;
      }
      default:
         MFEM_ABORT("Chebyshev smoother not implemented for order = " << order);
   }
}

void VoxelChebyshev::Mult(const Vector &x, Vector &y) const
{
   r = x;
   z.SetSize(x.Size());

   y = 0.0;
   for (int k = 0; k < order; ++k)
   {
      // Apply
      if (k > 0)
      {
         op.Mult(r, z);
         r = z;
      }

      // Scale residual by inverse diagonal
      block_jacobi.Mult(r, z);
      r = z;

      y.Add(coeffs[k], r);
   }
}

}
