//  Copyright 2024 PEI Weicheng
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/polynomial/hexahedron.hpp"

#include "mini/spatial/viscosity.hpp"
#include "mini/spatial/fem.hpp"
#include "mini/spatial/dg/lobatto.hpp"
#include "mini/spatial/fr/general.hpp"
#include "mini/spatial/fr/lobatto.hpp"
#include "mini/basis/vincent.hpp"
#include "mini/input/path.hpp"  // defines PROJECT_BINARY_DIR

#include "test/mesh/part.hpp"

using Projection = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents, true>;
using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;

auto case_name = PROJECT_BINARY_DIR + std::string("/test/mesh/double_mach");

class TestSpatialViscosity : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetRiemann();
  }
};
TEST_F(TestSpatialViscosity, LobattoFR) {
  auto part = Part(case_name, i_core, n_core);
  using Spatial = mini::spatial::fr::Lobatto<Part>;
  auto spatial = Spatial(&part);
  using Viscosity = mini::spatial::EnergyBasedViscosity<Part>;
  auto viscosity = Viscosity(&spatial);
  auto matrices = viscosity.BuildDampingMatrices();
}

// mpirun -n 4 ./part must be run in ../mesh
// mpirun -n 4 ./viscosity
int main(int argc, char* argv[]) {
  return Main(argc, argv);
}
