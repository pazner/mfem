#include "mfem.hpp"
#include "kershaw.hpp"
#include "fem/picojson.h"
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;

Mesh CreateKershawMesh(int N, double eps)
{
   Mesh mesh = Mesh::MakeCartesian3D(N, N, N, Element::HEXAHEDRON);
   KershawTransformation kt(mesh.Dimension(), eps, eps);
   mesh.Transform(kt);
   return mesh;
}

picojson::object RunIt(Solver *solver, int nsolves, int p, const std::string &name)
{
   picojson::object obj;

   const int n = solver->Height();
   Vector X(n), B(n);
   B.Randomize(0);

   MFEM_DEVICE_SYNC;
   tic_toc.Clear();
   tic_toc.Start();

   if (auto *dgminv = dynamic_cast<DGMassInverse*>(solver))
   {
      dgminv->Update();
   }
   else if (auto *dgminv_direct = dynamic_cast<DGMassInverse_Direct*>(solver))
   {
      dgminv_direct->Setup();
   }

   MFEM_DEVICE_SYNC;
   tic_toc.Stop();
   std::cout << "Setup done. " << std::flush;
   const double setup = tic_toc.RealTime();
   tic_toc.Clear();

   MFEM_DEVICE_SYNC;
   tic_toc.Clear();
   tic_toc.Start();
   for (int i = 0; i < nsolves; ++i)
   {
      X = 0.0;
      solver->Mult(B, X);
   }

   MFEM_DEVICE_SYNC;
   tic_toc.Stop();
   const double solve_total = tic_toc.RealTime();
   const double solve_each = solve_total / double(nsolves);
   const double total = setup + solve_total;
   std::cout << "Done.\n"
             << "Setup:         " << setup << '\n'
             << "Solve (each):  " << solve_each << '\n'
             << "Solve (total): " << solve_total << '\n'
             << "Total:         " << total << '\n' << std::endl;

   obj["setup"] = picojson::value(setup);
   obj["solve_each"] = picojson::value(solve_each);
   obj["solve_total"] = picojson::value(solve_total);
   obj["total"] = picojson::value(total);
   obj["p"] = picojson::value(double(p));
   obj["name"] = picojson::value(name);

   return obj;
}

int main(int argc, char *argv[])
{
   const char *device_config = "cpu";
   int nsolves = 100;
   // int ndofs = 2e6;
   int ndofs = 1800000;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&nsolves, "-ns", "--nsolves", "Number of mass solves.");
   args.AddOption(&ndofs, "-nd", "--ndofs", "Number of DOFs (approx).");
   args.ParseCheck();

   Device device(device_config);
   device.Print();

   picojson::array json;

   for (int p = 1; p < 9; ++p)
   {
      std::cout << "=== p = " << p << " ===\n";
      const int N = cbrt(ndofs)/double(p+1);
      Mesh mesh = CreateKershawMesh(N, 0.5);

      L2_FECollection fec(p, 3, BasisType::GaussLobatto);
      FiniteElementSpace fes(&mesh, &fec);
      std::cout << "# DOFS: " << fes.GetTrueVSize() << "\n\n";
      const IntegrationRule &ir = IntRules.Get(Geometry::CUBE, 2*p);

      std::unique_ptr<Solver> solver;

      solver.reset();
      solver.reset(new DGMassInverse(fes, ir, BasisType::GaussLegendre));
      std::cout << "Local CG Legendre... " << std::flush;
      json.push_back(picojson::value(RunIt(solver.get(), nsolves, p, "cg")));

      if (p < 8)
      {
         std::cout << "Explicit inverse... " << std::flush;
         solver.reset();
         solver.reset(new DGMassInverse_Direct(fes, ir, BatchSolverMode::CUBLAS));
         json.push_back(picojson::value(RunIt(solver.get(), nsolves, p, "inverse")));
      }

      std::cout << "Cholesky... " << std::flush;
      solver.reset();
      solver.reset(new DGMassInverse_Direct(fes, ir, BatchSolverMode::CUSOLVER));
      json.push_back(picojson::value(RunIt(solver.get(), nsolves, p, "cholesky")));
   }

   std::ofstream f("solver_results.json");
   f << picojson::value(json) << '\n';

   return 0;
}
