#ifndef MFEM_ELASTICITY_KERNEL_HELPERS_HPP
#define MFEM_ELASTICITY_KERNEL_HELPERS_HPP

#include "general/forall.hpp"
#include "linalg/tensor.hpp"

namespace mfem
{

using namespace mfem::internal;

namespace KernelHelpers
{

// MFEM_SHARED_3D_BLOCK_TENSOR definition
// Should be moved in backends/cuda/hip header files.
#if defined(__CUDA_ARCH__)
#define MFEM_SHARED_3D_BLOCK_TENSOR(name,T,bx,by,bz,...)\
MFEM_SHARED tensor<T,bx,by,bz,__VA_ARGS__> name;\
name(threadIdx.x, threadIdx.y, threadIdx.z) = tensor<T,__VA_ARGS__> {};
#else
#define MFEM_SHARED_3D_BLOCK_TENSOR(name,...) tensor<__VA_ARGS__> name {};
#endif

// Kernel helper functions
inline void CheckMemoryRestriction(int d1d, int q1d)
{
   MFEM_VERIFY(d1d <= q1d,
               "There should be more or equal quadrature points "
               "as there are dofs");
   MFEM_VERIFY(d1d <= MAX_D1D,
               "Maximum number of degrees of freedom in 1D reached."
               "This number can be increased globally in general/forall.hpp if "
               "device memory allows.");
   MFEM_VERIFY(q1d <= MAX_Q1D, "Maximum quadrature points 1D reached."
               "This number can be increased globally in "
               "general/forall.hpp if device memory allows.");
}

/**
 * @brief Multi-component gradient evaluation from DOFs to quadrature points in
 * reference coordinates.
 *
 * @note DeviceTensor<2> means RANK=2
 *
 * @tparam dim
 * @tparam d1d
 * @tparam q1d
 * @param B
 * @param G
 * @param smem
 * @param U
 * @param dUdxi
 */
template <int dim, int d1d, int q1d>
static inline MFEM_HOST_DEVICE void
CalcGrad(const tensor<double, q1d, d1d> &B, // q1d x d1d
         const tensor<double, q1d, d1d> &G, // q1d x d1d
         tensor<double,2,3,q1d,q1d,q1d> &smem,
         const DeviceTensor<4, const double> &U, // d1d x d1d x d1d x dim
         tensor<double, q1d, q1d, q1d, dim, dim> &dUdxi)
{
   for (int c = 0; c < dim; ++c)
   {
      MFEM_FOREACH_THREAD(dz,z,d1d)
      {
         MFEM_FOREACH_THREAD(dy,y,d1d)
         {
            MFEM_FOREACH_THREAD(dx,x,d1d)
            {
               smem(0,0,dx,dy,dz) = U(dx,dy,dz,c);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,d1d)
      {
         MFEM_FOREACH_THREAD(dy,y,d1d)
         {
            MFEM_FOREACH_THREAD(qx,x,q1d)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < d1d; ++dx)
               {
                  const double input = smem(0,0,dx,dy,dz);
                  u += input * B(qx,dx);
                  v += input * G(qx,dx);
               }
               smem(0,1,dz,dy,qx) = u;
               smem(0,2,dz,dy,qx) = v;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,d1d)
      {
         MFEM_FOREACH_THREAD(qy,y,q1d)
         {
            MFEM_FOREACH_THREAD(qx,x,q1d)
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int dy = 0; dy < d1d; ++dy)
               {
                  u += smem(0,2,dz,dy,qx) * B(qy,dy);
                  v += smem(0,1,dz,dy,qx) * G(qy,dy);
                  w += smem(0,1,dz,dy,qx) * B(qy,dy);
               }
               smem(1,0,dz,qy,qx) = u;
               smem(1,1,dz,qy,qx) = v;
               smem(1,2,dz,qy,qx) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,q1d)
      {
         MFEM_FOREACH_THREAD(qy,y,q1d)
         {
            MFEM_FOREACH_THREAD(qx,x,q1d)
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int dz = 0; dz < d1d; ++dz)
               {
                  u += smem(1,0,dz,qy,qx) * B(qz,dz);
                  v += smem(1,1,dz,qy,qx) * B(qz,dz);
                  w += smem(1,2,dz,qy,qx) * G(qz,dz);
               }
               dUdxi(qz,qy,qx,c,0) += u;
               dUdxi(qz,qy,qx,c,1) += v;
               dUdxi(qz,qy,qx,c,2) += w;
            }
         }
      }
      MFEM_SYNC_THREAD;
   } // vdim
}

/**
 * @brief Multi-component transpose gradient evaluation from DOFs to quadrature
 * points in reference coordinates with contraction of the D vector.
 *
 * @tparam dim
 * @tparam d1d
 * @tparam q1d
 * @param B
 * @param G
 * @param smem
 * @param U
 * @param F
 */
template <int dim, int d1d, int q1d>
static inline MFEM_HOST_DEVICE void
CalcGradTSum(const tensor<double, q1d, d1d> &B,
             const tensor<double, q1d, d1d> &G,
             tensor<double, 2, 3, q1d, q1d, q1d> &smem,
             const tensor<double, q1d, q1d, q1d, dim, dim> &U, // q1d x q1d x q1d x dim
             DeviceTensor<4, double> &F)                       // d1d x d1d x d1d x dim
{
   for (int c = 0; c < dim; ++c)
   {
      MFEM_FOREACH_THREAD(qz, z, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               for (int qx = 0; qx < q1d; ++qx)
               {
                  u += U(qx, qy, qz, 0, c) * G(qx, dx);
                  v += U(qx, qy, qz, 1, c) * B(qx, dx);
                  w += U(qx, qy, qz, 2, c) * B(qx, dx);
               }
               smem(0, 0, qz, qy, dx) = u;
               smem(0, 1, qz, qy, dx) = v;
               smem(0, 2, qz, qy, dx) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz, z, q1d)
      {
         MFEM_FOREACH_THREAD(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               for (int qy = 0; qy < q1d; ++qy)
               {
                  u += smem(0, 0, qz, qy, dx) * B(qy, dy);
                  v += smem(0, 1, qz, qy, dx) * G(qy, dy);
                  w += smem(0, 2, qz, qy, dx) * B(qy, dy);
               }
               smem(1, 0, qz, dy, dx) = u;
               smem(1, 1, qz, dy, dx) = v;
               smem(1, 2, qz, dy, dx) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz, z, d1d)
      {
         MFEM_FOREACH_THREAD(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               for (int qz = 0; qz < q1d; ++qz)
               {
                  u += smem(1, 0, qz, dy, dx) * B(qz, dz);
                  v += smem(1, 1, qz, dy, dx) * B(qz, dz);
                  w += smem(1, 2, qz, dy, dx) * G(qz, dz);
               }
               const double sum = u + v + w;
               F(dx, dy, dz, c) += sum;
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

/**
 * @brief Compute the gradient of all shape functions.
 *
 * @note TODO: Does not make use of shared memory on the GPU.
 *
 * @tparam dim
 * @tparam d1d
 * @tparam q1d
 * @param qx
 * @param qy
 * @param qz
 * @param B
 * @param G
 * @param invJ
 */
template <int dim, int d1d, int q1d> static inline MFEM_HOST_DEVICE
tensor<double, d1d, d1d, d1d, dim>
GradAllPhis(int qx, int qy, int qz,
            const tensor<double, q1d, d1d> &B,
            const tensor<double, q1d, d1d> &G,
            const tensor<double, dim, dim> &invJ)
{
   tensor<double, d1d, d1d, d1d, dim> dphi_dx;
   // G (x) B (x) B
   // B (x) G (x) B
   // B (x) B (x) G
   for (int dx = 0; dx < d1d; dx++)
   {
      for (int dy = 0; dy < d1d; dy++)
      {
         for (int dz = 0; dz < d1d; dz++)
         {
            dphi_dx(dx,dy,dz) =
               transpose(invJ) *
               tensor<double, dim> {G(qx, dx) * B(qy, dy) * B(qz, dz),
                                    B(qx, dx) * G(qy, dy) * B(qz, dz),
                                    B(qx, dx) * B(qy, dy) * G(qz, dz)
                                   };
         }
      }
   }
   return dphi_dx;
}
} // namespace KernelHelpers
}
#endif