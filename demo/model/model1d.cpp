// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cmath>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>

#include "mini/element/data.hpp"
#include "mini/mesh/dim1.hpp"
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/euler/hllc.hpp"
#include "mini/riemann/euler/ausm.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/riemann/rotated/single.hpp"
#include "mini/riemann/rotated/double.hpp"
#include "mini/riemann/rotated/burgers.hpp"
#include "mini/model/godunov1d.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace model {

template <class Riemann>
class SingleWave1DTest {
 public:
  explicit SingleWave1DTest(char** argv)
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
    auto model = Model(model_name_);
    model.ReadMesh(test_data_dir_ + mesh_name_);
    // Set Boundary Conditions:
    constexpr auto eps = 1e-5;
    model.SetBoundaryName("left", [&](Node& node) {
      return std::abs(node.X() + 0.5) < eps;
    });
    model.SetBoundaryName("right", [&](Node& node) {
      return std::abs(node.X() - 0.5) < eps;
    });
    model.SetFreeBoundary("left");
    model.SetFreeBoundary("right");
    // model.SetPeriodicBoundary("left", "right");
    model.SetInitialState([&](Cell& cell) {
      auto x = cell.Center().X();
      // cell.data.state = std::sin(x * acos(0.0) * 2);
      if (x < 0.0) {
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
  struct NodeData : public mesh::Empty {
    Flux flux;
  };
  struct CellData : public mesh::Data<
      double, 1/* dims */, 1/* scalars */, 0/* vectors */> {
   public:
    State state;
    void Write() {
      scalars[0] = state;
    }
  };
  using Mesh = mesh::Mesh<double, NodeData, CellData>;
  using Cell = typename Mesh::Cell;
  using Node = typename Mesh::Node;
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
class DoubleWave1DTest {
 public:
  explicit DoubleWave1DTest(char** argv)
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
    auto model = Model(model_name_);
    model.ReadMesh(test_data_dir_ + mesh_name_);
    // Set Boundary Conditions:
    constexpr auto eps = 1e-5;
    model.SetBoundaryName("left", [&](Node& node) {
      return std::abs(node.X() + 0.5) < eps;
    });
    model.SetBoundaryName("right", [&](Node& node) {
      return std::abs(node.X() - 0.5) < eps;
    });
    model.SetPeriodicBoundary("left", "right");
    model.SetInitialState([&](Cell& cell) {
      auto x = cell.Center().X();
      auto pi = std::acos(-1);
      cell.data.state[0] = std::sin(x * pi * 4);
      cell.data.state[1] = std::cos(x * pi * 4);
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
  struct NodeData : public mesh::Empty {
    Flux flux;
  };
  struct CellData : public mesh::Data<
      double, 1/* dims */, 2/* scalars */, 0/* vectors */> {
   public:
    State state;
    void Write() {
      scalars[0] = state[0];
      scalars[1] = state[1];
    }
  };
  using Mesh = mesh::Mesh<double, NodeData, CellData>;
  using Cell = typename Mesh::Cell;
  using Node = typename Mesh::Node;
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
class Euler1DTest {
 protected:
  // Types (Riemann related):
  using Gas = typename Riemann::Gas;
  using State = typename Riemann::State;
  using Flux = typename Riemann::Flux;
  // Types (Mesh related):
  struct NodeData : public mesh::Empty {
    Flux flux;
  };
  struct CellData : public mesh::Data<
      double, 2/* dims */, 2/* scalars */, 1/* vectors */> {
   public:
    State state;
    void Write() {
      auto primitive = Gas::ConservativeToPrimitive(state);
      scalars[0] = primitive.rho();
      scalars[1] = primitive.p();
      vectors[0][0] = primitive.u();
      vectors[0][1] = primitive.v();
    }
  };
  using Mesh = mesh::Mesh<double, NodeData, CellData>;
  using Cell = typename Mesh::Cell;
  using Node = typename Mesh::Node;
  using Model = model::Godunov<Mesh, Riemann>;

 public:
  explicit Euler1DTest(char** argv)
      : model_name_{argv[1]},
        mesh_name_{argv[2]},
        start_{std::atof(argv[3])},
        stop_{std::atof(argv[4])},
        n_steps_{std::atoi(argv[5])},
        output_rate_{std::atoi(argv[6])} {
    duration_ = stop_ - start_;
  }
  void Run() {
    Cell::scalar_names.at(0) = "rho";
    Cell::scalar_names.at(1) = "p";
    Cell::vector_names.at(0) = "u";
    auto model = Model(model_name_);
    model.ReadMesh(test_data_dir_ + mesh_name_);
    // Set Boundary Conditions:
    constexpr auto eps = 1e-5;
    model.SetBoundaryName("left", [&](Node& node) {
      return std::abs(node.X() + 0.5) < eps;
    });
    model.SetBoundaryName("right", [&](Node& node) {
      return std::abs(node.X() - 0.5) < eps;
    });
    model.SetFreeBoundary("left");
    model.SetFreeBoundary("right");
    // model.SetSolidBoundary("left");
    // model.SetSolidBoundary("right");
    // Set Initial Conditions:
    // TODO(PVC): move to main()
    /* Sod */
    // auto  left = State{1.00, 0, 0, 1.0};
    // auto right = State{.125, 0, 0, 0.1};

    /* Collision (start=0, stop=0.04, steps=2000)
    auto  left = State{5.99924, 19.59750, 0, 460.894};
    auto right = State{5.99242, -6.19633, 0, 46.0950};
     */
    /* Lax (start=0, stop=0.02, steps=2000)
    auto  left = State{1, 0, 0, 1000};
    auto right = State{1, 0, 0, 0.01};
     */
    /* Almost vacuumed  (start=0, stop=0.2, steps=2000)
    auto  left = State{1, -2, 0, 0.4};
    auto right = State{1, +2, 0, 0.4};
     */
    auto  left = State{3.857143, 2.629369, 0, 10.333333};
    Gas::PrimitiveToConservative(&left);
    // Gas::PrimitiveToConservative(&right);
    model.SetInitialState([&](Cell& cell) {
      auto x = cell.Center().X();
      if (x < -0.4) {
        cell.data.state = left;
      } else {
        auto right = State{1+0.2*std::sin(50*x), 0, 0, 1};
        Gas::PrimitiveToConservative(&right);
        cell.data.state = right;
      }
    });
    // Clean output directory:
    model.SetTimeSteps(duration_, n_steps_, output_rate_);
    auto output_dir = std::string("result/demo/euler/line/") + model_name_;
    model.SetOutputDir(output_dir + "/");
    system(("rm -rf " + output_dir).c_str());
    system(("mkdir -p " + output_dir).c_str());
    // Commit the calculation:
    model.Calculate();
  }

 protected:
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
    using LinearTest = mini::model::SingleWave1DTest<Single>;
    using Double = mini::riemann::rotated::Double;
    using DoubleLinearTest = mini::model::DoubleWave1DTest<Double>;
    using Burgers = mini::riemann::rotated::Burgers;
    using BurgersTest = mini::model::SingleWave1DTest<Burgers>;
    using Gas = mini::riemann::euler::IdealGas<1, 4>;
    using Riemann = mini::riemann::euler::Ausm<Gas, 2>;
    using Rotated = mini::riemann::rotated::Euler<Riemann>;
    using Euler1DTest = mini::model::Euler1DTest<Rotated>;
    if (std::strcmp(argv[1], "linear") == 0) {
      auto model = LinearTest(argv);
      model.Run();
    } else if (std::strcmp(argv[1], "burgers") == 0) {
      auto model = BurgersTest(argv);
      model.Run();
    } else if (std::strcmp(argv[1], "double") == 0) {
      auto model = DoubleLinearTest(argv);
      model.Run();
    } else if (std::strcmp(argv[1], "euler") == 0) {
      auto model = Euler1DTest(argv);
      model.Run();
    } else {
    }
  }
}
