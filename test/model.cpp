// Copyright 2019 Weicheng Pei and Minghao Yang

#include <string>

#include "gtest/gtest.h"

#include "mini/mesh/data.hpp"
#include "mini/mesh/dim2.hpp"
#include "mini/mesh/vtk.hpp"
#include "data.hpp"  // defines TEST_DATA_DIR
#include "mini/riemann/linear.hpp"
#include "mini/model/fvm.hpp"

namespace mini {

class SingleWaveModelTest : public :: testing::Test {
 protected:
  using Data = mesh::Data<double, 2/* dims */, 1/* scalars */, 0/* vectors */>;
  using Mesh = mesh::Mesh<double, /* Node */mesh::Empty,
                          /* Boundary */Data, /* Domain */Data>;
  using Domain = Mesh::Domain;
  using Boundary = Mesh::Boundary;
  using Riemann = riemann::SingleWave;
  using State = Riemann::State;
  using Model = model::FVM<Mesh, Riemann>;

 public:
  static std::string file_name;
};
std::string SingleWaveModelTest::file_name;
TEST_F(SingleWaveModelTest, SingleStep) {
  auto riemann = Riemann{/* speed */0.5};
  auto u_l = State{-1.0};
  auto u_r = State{+1.0};
  auto model = Model(riemann);
  model.ReadMesh(TEST_DATA_DIR + file_name);
  model.SetInitialState([&](Domain& domain) {
    if (domain.Center().X() < 0) {
      domain.data.scalars[0] = u_l;
    } else {
      domain.data.scalars[0] = u_r;
    }
  });
  model.SetWallBoundary([&](Boundary& boundary) {
  });
  model.SetTimeSteps(/* start */0.0, /* stop */1.0, /* start */1);
  model.Calculate();
}

}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc == 1) {
    mini::SingleWaveModelTest::file_name = "tiny.vtu";
  } else {
    mini::SingleWaveModelTest::file_name = argv[1];
  }
  return RUN_ALL_TESTS();
}
