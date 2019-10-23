// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cmath>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>

#include "mini/mesh/data.hpp"
#include "mini/mesh/dim2.hpp"
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/model/godunov.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace model {

template <class Riemann>
class Tube {
 protected:
  // Types (Riemann related):
  using Gas = typename Riemann::Gas;
  using State = typename Riemann::State;
  using Flux = typename Riemann::Flux;
  // Types (Mesh related):
  using NodeData = mesh::Empty;
  struct WallData : public mesh::Empty {
    Flux flux;
    Riemann riemann;
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
  using Mesh = mesh::Mesh<double, NodeData, WallData, CellData>;
  using Cell = typename Mesh::Cell;
  using Wall = typename Mesh::Wall;
  using Model = model::Godunov<Mesh, Riemann>;

 public:
  explicit Tube(char** argv)
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
    model.SetBoundaryName("left", [&](Wall& wall) {
      return std::abs(wall.Center().X() + 0.5) < eps;
    });
    model.SetBoundaryName("right", [&](Wall& wall) {
      return std::abs(wall.Center().X() - 0.5) < eps;
    });
    model.SetBoundaryName("top", [&](Wall& wall) {
      return std::abs(wall.Center().Y() - 0.01) < eps;
    });
    model.SetBoundaryName("bottom", [&](Wall& wall) {
      return std::abs(wall.Center().Y() + 0.01) < eps;
    });
    model.SetPeriodicBoundary("top", "bottom");
    model.SetFreeBoundary("left");
    model.SetFreeBoundary("right");
    // Set Initial Conditions:
    // TODO: move to main()
    /* Sod */
    auto  left = State{1.00, 0, 0, 1.0};
    auto right = State{.125, 0, 0, 0.1};
    /* Almost vaccumed
    auto  left = State{1, -2, 0, 0.4};
    auto right = State{1, +2, 0, 0.4};
     */
    Gas::PrimitiveToConservative(&left);
    Gas::PrimitiveToConservative(&right);
    model.SetInitialState([&](Cell& cell) {
      auto x = cell.Center().X();
      if (x < -0.0) {
        cell.data.state = left;
      } else {
        cell.data.state = right;
      }
    });
    // Clean output directory:
    model.SetTimeSteps(duration_, n_steps_, output_rate_);
    auto output_dir = std::string("result/demo/euler/tube/") + model_name_;
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
  if (argc == 7) {
    using Gas = mini::riemann::euler::IdealGas<1, 4>;
    using Riemann = mini::riemann::euler::Exact<Gas, 2>;
    using Rotated = mini::riemann::rotated::Euler<Riemann>;
    using Tube = mini::model::Tube<Rotated>;
    Tube(argv).Run();
  } else {
    std::cout << "usage: tube ";
    std::cout << "<sod|vacuum> ";
    std::cout << "<mesh> ";
    std::cout << "<start> <stop> <steps> ";
    std::cout << "<output_rate> ";
    std::cout << std::endl;
  }
}
