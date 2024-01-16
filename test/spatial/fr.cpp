//  Copyright 2023 PEI Weicheng
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "mpi.h"
#include "gtest/gtest.h"
#include "gtest_mpi/gtest_mpi.hpp"

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/mesh/vtk.hpp"
#include "mini/limiter/weno.hpp"
#include "mini/riemann/concept.hpp"
#include "mini/riemann/rotated/multiple.hpp"
#include "mini/riemann/diffusive/linear.hpp"
#include "mini/riemann/diffusive/direct_dg.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/spatial/fr/general.hpp"
#include "mini/spatial/fr/lobatto.hpp"
#include "mini/basis/vincent.hpp"
#include "mini/input/path.hpp"  // defines PROJECT_BINARY_DIR

constexpr int kComponents{2}, kDimensions{3}, kDegrees{2};
using Scalar = double;
using Convection = mini::
    riemann::rotated::Multiple<Scalar, kComponents, kDimensions>;
using Diffusion = mini::riemann::diffusive::DirectDG<
    mini::riemann::diffusive::Isotropic<Scalar, kComponents>
>;
using Riemann = mini::riemann::ConvectionDiffusion<Convection, Diffusion>;
using Coord = typename Riemann::Vector;
using Value = typename Riemann::Conservative;
Value func(const Coord& xyz) {
  auto r = std::hypot(xyz[0] - 2, xyz[1] - 0.5);
  return Value(r, 1 - r + (r >= 1));
}
Value moving(const Coord& xyz, double t) {
  auto x = xyz[0], y = xyz[1];
  return Value(x + y, x - y);
}
using Gx = mini::gauss::Lobatto<Scalar, kDegrees + 1>;
using Projection = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents, true>;
using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;

int n_core, i_core;
double time_begin;

auto case_name = PROJECT_BINARY_DIR + std::string("/test/mesh/double_mach");

class TestSpatialFR : public ::testing::Test {
 protected:
  void SetUp() override;
};
void TestSpatialFR::SetUp() {
  using Jacobian = typename Riemann::Jacobian;
  Riemann::SetConvectionCoefficient(
    Jacobian{ {3., 0.}, {0., 4.} },
    Jacobian{ {5., 0.}, {0., 6.} },
    Jacobian{ {7., 0.}, {0., 8.} }
  );
  Riemann::SetDiffusionCoefficient(1.0);
  Riemann::SetBetaValues(2.0, 1.0 / 12);
}
template <typename Spatial>
auto GetResidualColumn(Spatial &spatial, Part &part) {
  spatial.SetSmartBoundary("4_S_31", moving);  // Left
  spatial.SetSolidWall("4_S_1");   // Back
  spatial.SetSubsonicInlet("4_S_32", moving);  // Front
  spatial.SetSubsonicOutlet("4_S_23", moving);  // Right
  spatial.SetSupersonicInlet("4_S_27", moving);  // Top
  spatial.SetSupersonicOutlet("4_S_15");  // Gap
  spatial.SetSupersonicOutlet("4_S_19");  // Bottom
  for (auto *cell_ptr : part.GetLocalCellPointers()) {
    cell_ptr->Approximate(func);
  }
  spatial.SetTime(1.5);
  std::printf("%s() proc[%d/%d] cost %f sec\n",
      spatial.name(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  auto column = spatial.GetSolutionColumn();
  assert(column.size() == part.GetCellDataSize());
  spatial.SetSolutionColumn(column);
  std::printf("solution.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  column = spatial.GetResidualColumn();
  std::printf("residual.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);
  return column;
}
TEST_F(TestSpatialFR, CompareResiduals) {
  auto part = Part(case_name, i_core, n_core);
  /* aproximated by Lagrange basis on Lobatto roots with general correction functions */
  time_begin = MPI_Wtime();
  using General = mini::spatial::fr::General<Part>;
  using Vincent = mini::basis::Vincent<Scalar>;
  auto general = General(&part, Vincent::HuynhLumpingLobatto(kDegrees));
  auto general_residual = GetResidualColumn(general, part);
  /* aproximated by Lagrange basis on Lobatto roots with Huynh's correction functions */
  time_begin = MPI_Wtime();
  using Lobatto = mini::spatial::fr::Lobatto<Part>;
  auto lobatto = Lobatto(&part);
  auto lobatto_residual = GetResidualColumn(lobatto, part);
  EXPECT_NEAR(0, (general_residual - lobatto_residual).norm(), 1e-15);
}

// mpirun -n 4 ./part must be run in ../mesh
// mpirun -n 4 ./fr
int main(int argc, char* argv[]) {
  // Initialize MPI before any call to gtest_mpi
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &n_core);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  // Intialize google test
  ::testing::InitGoogleTest(&argc, argv);

  // Add a test environment, which will initialize a test communicator
  // (a duplicate of MPI_COMM_WORLD)
  ::testing::AddGlobalTestEnvironment(new gtest_mpi::MPITestEnvironment());

  auto& test_listeners = ::testing::UnitTest::GetInstance()->listeners();

  // Remove default listener and replace with the custom MPI listener
  delete test_listeners.Release(test_listeners.default_result_printer());
  test_listeners.Append(new gtest_mpi::PrettyMPIUnitTestResultPrinter());

  // run tests
  auto exit_code = RUN_ALL_TESTS();

  // Finalize MPI before exiting
  MPI_Finalize();

  return exit_code;
}
