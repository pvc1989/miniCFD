// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cmath>
#include <string>

#include "gtest/gtest.h"

#include "mini/mesh/data.hpp"
#include "mini/mesh/dim2.hpp"
#include "mini/mesh/vtk.hpp"
#include "data.hpp"  // defines TEST_DATA_DIR
#include "mini/riemann/linear.hpp"
#include "mini/riemann/burgers.hpp"
#include "mini/model/single_wave.hpp"

namespace mini {
namespace model {

// class SingleWaveTest : public :: testing::Test {
//  protected:
//   using NodeData = mesh::Empty;
//   using WallData = mesh::Data<
//       double, 2/* dims */, 2/* scalars */, 0/* vectors */>;
//   using CellData = mesh::Data<
//       double, 2/* dims */, 1/* scalars */, 0/* vectors */>;
//   using Mesh = mesh::Mesh<double, NodeData, WallData, CellData>;
//   using Cell = Mesh::Cell;
//   using Wall = Mesh::Wall;
//   using Riemann = riemann::SingleWave;
//   using State = Riemann::State;
//   using Model = model::SingleWave<Mesh, Riemann>;

//  public:
//   static const char* file_name;
//   static double duration;
//   static int n_steps;
//   static int refresh_rate;
//  protected:
//   const std::string test_data_dir_{TEST_DATA_DIR};
// };
// const char* SingleWaveTest::file_name;
// double SingleWaveTest::duration;
// int SingleWaveTest::n_steps;
// int SingleWaveTest::refresh_rate;
// TEST_F(SingleWaveTest, SingleStep) {
//   Mesh::Cell::scalar_names.at(0) = "U";
//   auto u_l = State{-1.0};
//   auto u_r = State{+1.0};
//   auto model = Model(1.0, 0.0);
//   model.ReadMesh(test_data_dir_ + file_name);
//   model.SetInitialState([&](Cell& cell) {
//     if (cell.Center().X() < 0) {
//       cell.data.scalars[0] = u_l;
//     } else {
//       cell.data.scalars[0] = u_r;
//     }
//   });
//   model.SetTimeSteps(duration, n_steps, refresh_rate);
//   model.SetOutputDir("linear/");
//   model.Calculate();
// }

class BurgersTest : public :: testing::Test {
 protected:
  using NodeData = mesh::Empty;
  using WallData = mesh::Data<
      double, 2/* dims */, 1/* scalars */, 1/* vectors */>;
  using CellData = mesh::Data<
      double, 2/* dims */, 1/* scalars */, 0/* vectors */>;
  using Mesh = mesh::Mesh<double, NodeData, WallData, CellData>;
  using Cell = Mesh::Cell;
  using Wall = Mesh::Wall;
  using Riemann = riemann::Burgers;
  using State = Riemann::State;
  using Model = model::Burgers<Mesh, Riemann>;

 public:
  static const char* file_name;
  static double duration;
  static int n_steps;
  static int refresh_rate;
 protected:
  const std::string test_data_dir_{TEST_DATA_DIR};
  const double kOmega = std::acos(0);
};
const char* BurgersTest::file_name;
double BurgersTest::duration;
int BurgersTest::n_steps;
int BurgersTest::refresh_rate;
TEST_F(BurgersTest, SingleStep) {
  Mesh::Cell::scalar_names.at(0) = "U";
  auto model = Model();
  model.ReadMesh(test_data_dir_ + file_name);
  model.SetInitialState([&](Cell& cell) {
    auto x = cell.Center().X();
    auto y = cell.Center().Y();
    cell.data.scalars[0] = -std::sin(kOmega * x);
  });
  model.SetTimeSteps(duration, n_steps, refresh_rate);
  model.SetOutputDir("burgers/");
  model.Calculate();
}
}  // namespace model
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc == 1) {
    mini::model::BurgersTest::file_name = "medium.vtu";
    mini::model::BurgersTest::duration = 0.5;
    mini::model::BurgersTest::n_steps = 1000;
    mini::model::BurgersTest::refresh_rate = 10;
  } else {
    mini::model::BurgersTest::file_name = argv[1];
  }
  return RUN_ALL_TESTS();
}
