#include "mfem.hpp"
#include "mg_elasticity.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   int n = 100;

   OptionsParser args(argc, argv);
   args.AddOption(&n, "-n", "--n", "Width/height of the image.");
   args.ParseCheck();

   ofstream f("diag.pgm");

   f << "P2\n";
   f << n << " " << n << '\n';
   f << "255" << '\n';
   for (int i = 0; i < n; ++i)
   {
      for (int j = 0; j < n; ++j)
      {
         if (i < j) { f << "255\n"; }
         else { f << "0\n"; }
      }
   }

   return 0;
}
