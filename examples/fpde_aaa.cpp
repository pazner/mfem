//                               Implementation of the AAA algorithm
//
//
//
//
//
//
//
//
//
//
//   REFERENCE
//
//   Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). The AAA
//   algorithm for rational approximation. SIAM Journal on Scientific
//   Computing, 40(3), A1494-A1522.

#include "mfem.hpp"

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


void RationalApproximation_AAA(const Vector &val,
                               const Vector &pt,           // inputs
                               Array<double> &z, Array<double> &f, Vector &w, // outputs
                               double tol = 1.0e-8, int max_order = 100);


int main(int argc, char *argv[])
{


}


void RationalApproximation_AAA(const Vector &val, const Vector &pt,
                               Array<double> &z, Array<double> &f, Vector &w,
                               double tol, int max_order)
{
   MFEM_VERIFY(pt.Size() == val.Size(), "size mismatch");

   z.SetSize(0);
   f.SetSize(0);
   Vector f_vec;

   int size = val.Size();
   SparseMatrix SF(val);
   DenseMatrix C;
   Array<double> C_i;

   double R = val.Sum()/size;
   Array<int> J(size);
   for (int i = 0; i < size; i++)
   {
      J[i] = i;
   }



   for (int i = 0; i < max_order; i++)
   {
      // Vector tmp(val);
      // tmp -= R;
      int index = 0;
      double tmp_max = 0;
      for (int j = 0; j < size; j++)
      {
         double tmp = abs(val(j)-R);
         if (tmp > tmp_max)
         {
            tmp_max = tmp;
            index = j;
         }
      }
      z.Append(pt(index));
      f.Append(val(index));
      J.DeleteFirst(index);

      Array<double> C_tmp(size-1);
      int cnt = 0;
      for (int j = 0; j < size; j++)
      {
         if (j != index)
         {
            C_tmp[cnt] = 1.0/(pt(j)-pt(index));
         }
         cnt++;
      }
      C_i.Append(C_tmp);

      f_vec.SetDataAndSize(f.GetData(),f.Size());



   }



}