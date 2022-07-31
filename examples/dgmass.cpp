#include "mfem.hpp"

using namespace std;
using namespace mfem;

// 1D transformation at the right boundary.
double right(const double eps, const double x)
{
   return (x <= 0.5) ? (2-eps) * x : 1 + eps*(x-1);
}

// 1D transformation at the left boundary
double left(const double eps, const double x)
{
   return 1-right(eps,1-x);
}

// Transition from a value of "a" for x=0, to a value of "b" for x=1. Smoothness
// is controlled by the parameter "s", taking values 0, 1, or 2.
double step(const double a, const double b, double x, int s)
{
   if (x <= 0) { return a; }
   if (x >= 1) { return b; }
   switch (s)
   {
      case 0:
      default:
         return a + (b-a) * (x);
      case 1: return a + (b-a) * (x*x*(3-2*x));
      case 2: return a + (b-a) * (x*x*x*(x*(6*x-15)+10));
   }
}

// 3D version of a generalized Kershaw mesh transformation, see D. Kershaw,
// "Differencing of the diffusion equation in Lagrangian hydrodynamic codes",
// JCP, 39:375–395, 1981.
//
// The input mesh should be Cartesian nx x ny x nz with nx divisible by 6 and
// ny, nz divisible by 2.
//
// The eps parameters are in (0, 1]. Uniform mesh is recovered for epsy=epsz=1.
void kershaw(const double epsy, const double epsz, const int smoothness,
             const double x, const double y, const double z,
             double &X, double &Y, double &Z)
{
   X = x;

   int layer = x*6.0;
   double lambda = (x-layer/6.0)*6;

   // The x-range is split in 6 layers going from left-to-left, left-to-right,
   // right-to-left (2 layers), left-to-right and right-to-right yz-faces.
   switch (layer)
   {
      case 0:
         Y = left(epsy, y);
         Z = left(epsz, z);
         break;
      case 1:
      case 4:
         Y = step(left(epsy, y), right(epsy, y), lambda, smoothness);
         Z = step(left(epsz, z), right(epsz, z), lambda, smoothness);
         break;
      case 2:
         Y = step(right(epsy, y), left(epsy, y), lambda/2, smoothness);
         Z = step(right(epsz, z), left(epsz, z), lambda/2, smoothness);
         break;
      case 3:
         Y = step(right(epsy, y), left(epsy, y), (1+lambda)/2, smoothness);
         Z = step(right(epsz, z), left(epsz, z), (1+lambda)/2, smoothness);
         break;
      default:
         Y = right(epsy, y);
         Z = right(epsz, z);
         break;
   }
}

struct KershawTransformation : VectorCoefficient
{
   double epsy, epsz;
   int dim, s;
   KershawTransformation(int dim_, double epsy_, double epsz_, int s_=0)
      : VectorCoefficient(dim_), epsy(epsy_), epsz(epsz_), dim(dim_), s(s_) { }
   using VectorCoefficient::Eval;
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      double xyz[3];
      Vector transip(xyz, 3);
      T.Transform(ip, transip);
      if (dim == 1)
      {
         V[0] = xyz[0]; // no transformation in 1D
      }
      else if (dim == 2)
      {
         double z=0, zt;
         kershaw(epsy, epsz, s, xyz[0], xyz[1], z, V[0], V[1], zt);
      }
      else // dim == 3
      {
         kershaw(epsy, epsz, s, xyz[0], xyz[1], xyz[2], V[0], V[1], V[2]);
      }
   }
};

Mesh CreateKershawMesh(int N, double eps)
{
   Mesh mesh = Mesh::MakeCartesian3D(N, N, N, Element::HEXAHEDRON);
   KershawTransformation kt(mesh.Dimension(), eps, eps);
   mesh.Transform(kt);
   return mesh;
}

int main(int argc, char *argv[])
{
   int order = 1;
   int n = 10;
   double e = 0.5;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&n, "-n", "--nx", "Number of elements in x dimension");
   args.AddOption(&e, "-e", "--eps", "Mesh distortion");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.ParseCheck();

   Mesh mesh = CreateKershawMesh(n, e);
   DG_FECollection fec(order, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   DGMassInverse massinv(fes, BasisType::GaussLegendre);

   int ndofs = fes.GetTrueVSize();
   std::cout << "Problem size: " << ndofs << '\n';
   Vector X(ndofs), B(ndofs);
   X.Randomize(1);
   B = 0.0;

   massinv.Mult(B, X);
   return 0;
}
