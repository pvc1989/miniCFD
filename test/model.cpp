// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

#include "mini/mesh/data.hpp"
#include "mini/mesh/dim2.hpp"
#include "mini/mesh/vtk.hpp"
#include "mini/riemann/linear.hpp"
#include "mini/riemann/burgers.hpp"
#include "mini/model/single_wave.hpp"
#include "mini/model/double_wave.hpp"
#include "data.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace model {

template <class Riemann>
class SingleWaveTest {
 public:
  explicit SingleWaveTest(char** argv)
      : model_name_{argv[1]}, 
        mesh_name_{argv[2]},
        start_{std::atof(argv[3])},
        stop_{std::atof(argv[4])},
        n_steps_{std::atoi(argv[5])},
        output_rate_{std::atoi(argv[6])} { 
    duration_ = stop_ - start_;
  }

  void Run() {
    Mesh::Cell::scalar_names.at(0) = "U";
    auto u_l = State{-1.0};
    auto u_r = State{+1.0};
    auto model = Model(1.0, 0.0);
    model.ReadMesh(test_data_dir_ + mesh_name_);
    model.SetInitialState([&](Cell& cell) {
      if (cell.Center().X() < 0) {
        cell.data.scalars[0] = u_l;
      } else {
        cell.data.scalars[0] = u_r;
      }
    });
    model.SetTimeSteps(duration_, n_steps_, output_rate_);
    std::string command  = "rm -rf " + model_name_;
    system(command.c_str());
    command  = "mkdir " + model_name_;
    system(command.c_str());
    model.SetOutputDir(model_name_ + "/");
    model.Calculate();
  }

 protected:
  // Types:
  using NodeData = mesh::Empty;
  using WallData = mesh::Data<
      double, 2/* dims */, 2/* scalars */, 0/* vectors */>;
  using CellData = mesh::Data<
      double, 2/* dims */, 1/* scalars */, 0/* vectors */>;
  using Mesh = mesh::Mesh<double, NodeData, WallData, CellData>;
  using Cell = Mesh::Cell;
  using Wall = Mesh::Wall;
  using State = typename Riemann::State;
  using Model = model::SingleWave<Mesh, Riemann>;
  // Data:
  const std::string test_data_dir_{TEST_DATA_DIR};
  const std::string model_name_;
  const std::string mesh_name_;
  double duration_;
  double start_;
  double stop_;
  int n_steps_;
  int output_rate_;
};

template <class Riemann>
class DoubleWaveTest {
 public:
  explicit DoubleWaveTest(char** argv)
      : model_name_{argv[1]}, 
        mesh_name_{argv[2]},
        start_{std::atof(argv[3])},
        stop_{std::atof(argv[4])},
        n_steps_{std::atoi(argv[5])},
        output_rate_{std::atoi(argv[6])} { 
    duration_ = stop_ - start_;
  }

  void Run() {
    Mesh::Cell::scalar_names.at(0) = "U_0";
    Mesh::Cell::scalar_names.at(1) = "U_1";
    Mesh::Cell::scalar_names.at(2) = "U_2";
    auto u_l = State{5.0, 1.0};
    auto u_r = State{4.0, 3.0};
    auto a = Matrix{1, 0, 0, -1};
    auto b = Matrix{0, 0, 0, 0};
    auto model = Model(a, b);
    model.ReadMesh(test_data_dir_ + mesh_name_);
    model.SetInitialState([&](Cell& cell) {
      if (cell.Center().X() < 0) {
        cell.data.scalars[0] = u_l[0];
        cell.data.scalars[1] = u_l[1];
        cell.data.scalars[2] = u_l[0] - u_l[1];
      } else {
        cell.data.scalars[0] = u_r[0];
        cell.data.scalars[1] = u_r[1];
        cell.data.scalars[2] = u_r[0] - u_r[1];
      }
    });
    model.SetTimeSteps(duration_, n_steps_, output_rate_);
    std::string command  = "rm -rf " + model_name_;
    system(command.c_str());
    command  = "mkdir " + model_name_;
    system(command.c_str());
    model.SetOutputDir(model_name_ + "/");
    model.Calculate();
  }

 protected:
  // Types:
  using NodeData = mesh::Empty;
  using WallData = mesh::Data<
      double, 2/* dims */, 2/* scalars */, 2/* vectors */>;
  using CellData = mesh::Data<
      double, 2/* dims */, 3/* scalars */, 0/* vectors */>;
  using Mesh = mesh::Mesh<double, NodeData, WallData, CellData>;
  using Cell = Mesh::Cell;
  using Wall = Mesh::Wall;
  using State = typename Riemann::State;
  using Matrix = typename Riemann::Matrix;
  using Model = model::DoubleWave<Mesh, Riemann>;
  // Data:
  const std::string test_data_dir_{TEST_DATA_DIR};
  const std::string model_name_;
  const std::string mesh_name_;
  double duration_;
  double start_;
  double stop_;
  int n_steps_;
  int output_rate_;
};

}  // namespace model
}  // namespace mini

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::cout << "usage: model ";  // argv[0] == "model"
    std::cout << "<linear|burgers|doublelinear> ";
    std::cout << "<mesh> ";
    std::cout << "<start> <stop> <steps> ";
    std::cout << "<output_rate> ";
    std::cout << std::endl;
  } else if (argc == 7) {
    using LinearTest = mini::model::SingleWaveTest<mini::riemann::SingleWave>;
    using DoubleLinearTest = mini::model::DoubleWaveTest<mini::riemann::MultiWave<2>>;
    using BurgersTest = mini::model::SingleWaveTest<mini::riemann::Burgers>;
    if (std::strcmp(argv[1], "linear") == 0) {
      auto model = LinearTest(argv);
      model.Run();
    } else if (std::strcmp(argv[1], "burgers") == 0) {
      auto model = BurgersTest(argv);
      model.Run();
    } else if (std::strcmp(argv[1], "doublelinear") == 0) {
      auto model = DoubleLinearTest(argv);
      model.Run();
    }
  }
}
