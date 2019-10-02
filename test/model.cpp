// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cmath>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>

#include "mini/mesh/data.hpp"
#include "mini/mesh/dim2.hpp"
#include "mini/mesh/vtk.hpp"
#include "mini/riemann/rotated/single.hpp"
#include "mini/riemann/rotated/double.hpp"
#include "mini/riemann/rotated/burgers.hpp"
#include "mini/model/godunov.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

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
    auto u_l = State{+1.0};
    auto u_r = State{-1.0};
    auto model = Model();
    Riemann::global_coefficient[0] = 1.0;
    Riemann::global_coefficient[1] = 0.0;
    model.ReadMesh(test_data_dir_ + mesh_name_);
    model.SetBoundaryName("left", [&](Wall& wall) {
      return wall.Center().X() == -2.0;
    });
    model.SetBoundaryName("right", [&](Wall& wall) {
      return wall.Center().X() == +2.0;
    });
    model.SetBoundaryName("top", [&](Wall& wall) {
      return wall.Center().Y() == +1.0;
    });
    model.SetBoundaryName("bottom", [&](Wall& wall) {
      return wall.Center().Y() == -1.0;
    });
    // model.SetInletBoundary("left");
    // model.SetOutletBoundart("right");
    model.SetPeriodicBoundary("left", "right");
    model.SetPeriodicBoundary("top", "bottom");
    // model.SetSolidBoundary("left");
    // model.SetSolidBoundary("right");
    // model.SetFreeBoundary("top");
    // model.SetFreeBoundary("bottom");
    model.SetInitialState([&](Cell& cell) {
      auto x = cell.Center().X();
      auto y = cell.Center().Y();
      // cell.data.state = std::sin(x * acos(0.0)) *
      //                   std::sin(y * acos(0.0) * 2);
      if (x < -0.0) {
        cell.data.state = u_l;
      } else {
        cell.data.state = u_r;
      }
    });
    model.SetTimeSteps(duration_, n_steps_, output_rate_);
    auto output_dir = std::string("result/demo/") + model_name_;
    model.SetOutputDir(output_dir + "/");
    system(("rm -rf " + output_dir).c_str());
    system(("mkdir -p " + output_dir).c_str());
    // Commit the calculation:
    model.Calculate();
  }

 protected:
  using Jacobi = typename Riemann::Jacobi;
  using State = typename Riemann::State;
  using Flux = typename Riemann::Flux;
  using Coefficient = algebra::Column<Jacobi, 2>;
  // Types:
  using NodeData = mesh::Empty;
  struct WallData : public mesh::Empty {
    Flux flux;
    Riemann riemann;
  };
  struct CellData : public mesh::Data<
      double, 2/* dims */, 1/* scalars */, 0/* vectors */> {
   public:
    State state;
    void Write() {
      scalars[0] = state;
    }
  };
  using Mesh = mesh::Mesh<double, NodeData, WallData, CellData>;
  using Cell = typename Mesh::Cell;
  using Wall = typename Mesh::Wall;
  using Model = model::Godunov<Mesh, Riemann>;
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
    auto u_l = State{1.0, 2.0};
    auto u_r = State{1.5, 2.5};
    auto model = Model();
    Riemann::global_coefficient[0] = Jacobi{{1, 0}, {0, -1}};
    Riemann::global_coefficient[1] = Jacobi{{1, 0}, {0, -1}};
    model.ReadMesh(test_data_dir_ + mesh_name_);
    model.SetBoundaryName("left", [&](Wall& wall) {
      return wall.Center().X() == -2.0;
    });
    model.SetBoundaryName("right", [&](Wall& wall) {
      return wall.Center().X() == +2.0;
    });
    model.SetBoundaryName("top", [&](Wall& wall) {
      return wall.Center().Y() == +1.0;
    });
    model.SetBoundaryName("bottom", [&](Wall& wall) {
      return wall.Center().Y() == -1.0;
    });
    // model.SetInletBoundary("left");
    // model.SetOutletBoundart("right");
    model.SetPeriodicBoundary("left", "right");
    model.SetPeriodicBoundary("top", "bottom");
    // model.SetSolidBoundary("left");
    // model.SetSolidBoundary("right");
    // model.SetFreeBoundary("top");
    // model.SetFreeBoundary("bottom");
    model.SetInitialState([&](Cell& cell) {
      auto x = cell.Center().X();
      if (x < -0.0) {
        cell.data.state = u_l;
      } else {
        cell.data.state = u_r;
      }
    });
    model.SetTimeSteps(duration_, n_steps_, output_rate_);
    auto output_dir = std::string("result/demo/") + model_name_;
    model.SetOutputDir(output_dir + "/");
    system(("rm -rf " + output_dir).c_str());
    system(("mkdir -p " + output_dir).c_str());
    // Commit the calculation:
    model.Calculate();
  }

 protected:
  using Jacobi = typename Riemann::Jacobi;
  using State = typename Riemann::State;
  using Flux = typename Riemann::Flux;
  // Types:
  using NodeData = mesh::Empty;
  struct WallData : public mesh::Empty {
    Flux flux;
    Riemann riemann;
  };
  struct CellData : public mesh::Data<
      double, 2/* dims */, 2/* scalars */, 0/* vectors */> {
   public:
    State state;
    void Write() {
      scalars[0] = state[0];
      scalars[1] = state[1];
    }
  };
  using Mesh = mesh::Mesh<double, NodeData, WallData, CellData>;
  using Cell = typename Mesh::Cell;
  using Wall = typename Mesh::Wall;
  using Model = model::Godunov<Mesh, Riemann>;
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
    std::cout << "<linear|burgers|double> ";
    std::cout << "<mesh> ";
    std::cout << "<start> <stop> <steps> ";
    std::cout << "<output_rate> ";
    std::cout << std::endl;
  } else if (argc == 7) {
    using Single = mini::riemann::rotated::Single;
    using LinearTest = mini::model::SingleWaveTest<Single>;
    using Double = mini::riemann::rotated::Double;
    using DoubleLinearTest = mini::model::DoubleWaveTest<Double>;
    using Burgers = mini::riemann::rotated::Burgers;
    using BurgersTest = mini::model::SingleWaveTest<Burgers>;
    if (std::strcmp(argv[1], "linear") == 0) {
      auto model = LinearTest(argv);
      model.Run();
    } else if (std::strcmp(argv[1], "burgers") == 0) {
      auto model = BurgersTest(argv);
      model.Run();
    } else if (std::strcmp(argv[1], "double") == 0) {
      auto model = DoubleLinearTest(argv);
      model.Run();
    } else {
    }
  }
}
