// Copyright 2019 Weicheng Pei and Minghao Yang

#include <string>

#include "gtest/gtest.h"

#include "mini/mesh/data.hpp"
#include "mini/mesh/dim2.hpp"
#include "mini/mesh/vtk.hpp"
#include "data.hpp"  // defines TEST_DATA_DIR
#include "mini/riemann/linear.hpp"
#include "mini/model/single_wave.hpp"

namespace mini {
namespace model {

class SingleWaveTest : public :: testing::Test {
 protected:
  using NodeData = mesh::Empty;
  using BoundaryData = mesh::Data<
      double, 2/* dims */, 2/* scalars */, 0/* vectors */>;
  using DomainData = mesh::Data<
      double, 2/* dims */, 1/* scalars */, 0/* vectors */>;
  using Mesh = mesh::Mesh<double, NodeData, BoundaryData, DomainData>;
  using Domain = Mesh::Domain;
  using Boundary = Mesh::Boundary;
  using Riemann = riemann::SingleWave;
  using State = Riemann::State;
  using Model = model::SingleWave<Mesh, Riemann>;

 public:
  static const char* file_name;
  static double duration;
  static int n_steps;
  static int refresh_rate;
 protected:
  const std::string test_data_dir_{TEST_DATA_DIR};
};
const char* SingleWaveTest::file_name;
double SingleWaveTest::duration;
int SingleWaveTest::n_steps;
int SingleWaveTest::refresh_rate;
TEST_F(SingleWaveTest, SingleStep) {
  Mesh::Domain::scalar_names.at(0) = "U";
  auto u_l = State{-1.0};
  auto u_r = State{+1.0};
  auto model = Model(1.0, 0.0);
  model.ReadMesh(test_data_dir_ + file_name);
  model.SetInitialState([&](Domain& domain) {
    if (domain.Center().X() < 0) {
      domain.data.scalars[0] = u_l;
    } else {
      domain.data.scalars[0] = u_r;
    }
  });
  model.SetWallBoundary([&](Boundary& boundary) {
  });
  model.SetTimeSteps(duration, n_steps, refresh_rate);
  model.SetOutputDir("result/");
  model.Calculate();
}

}  // namespace model
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc == 1) {
    mini::model::SingleWaveTest::file_name = "medium.vtu";
    mini::model::SingleWaveTest::duration = 1.0;
    mini::model::SingleWaveTest::n_steps = 50;
    mini::model::SingleWaveTest::refresh_rate = 1;
  } else {
    mini::model::SingleWaveTest::file_name = argv[1];
  }
  return RUN_ALL_TESTS();
}
