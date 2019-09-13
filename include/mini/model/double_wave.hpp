// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MODEL_DOUBLE_WAVE_HPP_
#define MINI_MODEL_DOUBLE_WAVE_HPP_

#include <cmath>
#include <memory>
#include <string>

#include "mini/riemann/linear.hpp"
#include "mini/mesh/data.hpp"
#include "mini/mesh/dim2.hpp"
#include "mini/mesh/vtk.hpp"

namespace mini {
namespace model {

template <class Mesh, class Riemann>
class DoubleWave {
  using Wall = typename Mesh::Wall;
  using Cell = typename Mesh::Cell;
  using State = typename Riemann::State;
  using Flux = typename Riemann::Flux;
  using Matrix = typename Riemann::Matrix;
  using VtkReader = typename mesh::VtkReader<Mesh>;
  using VtkWriter = typename mesh::VtkWriter<Mesh>;

 public:
  DoubleWave(Matrix a, Matrix b) : a_(a), b_(b) {}
  bool ReadMesh(std::string const& file_name) {
    reader_ = VtkReader();
    if (reader_.ReadFromFile(file_name)) {
      mesh_ = reader_.GetMesh();
      return true;
    } else {
      return false;
    }
  }
  // Mutators:
  template <class Visitor>
  void SetInitialState(Visitor&& visitor) {
    mesh_->ForEachCell(visitor);
  }
  void SetTimeSteps(double duration, int n_steps, int refresh_rate) {
    duration_ = duration;
    n_steps_ = n_steps;
    step_size_ = duration / n_steps;
    refresh_rate_ = refresh_rate;
  }
  void SetOutputDir(std::string dir) {
    dir_ = dir;
  }
  // Major computation:
  void Calculate() {
    writer_ = VtkWriter();
    auto filename = dir_ + std::to_string(0) + ".vtu";
    bool pass = OutputCurrentResult(filename);
    assert(pass);
    mesh_->ForEachWall([&](Wall& wall){
      double cos = (wall.Tail()->Y() - wall.Head()->Y()) / wall.Measure();
      double sin = (wall.Head()->X() - wall.Tail()->X()) / wall.Measure();
      wall.data.vectors[0] = {cos * a_[0][0] + sin * b_[0][0],
                              cos * a_[0][1] + sin * b_[0][1]};
      wall.data.vectors[1] = {cos * a_[1][0] + sin * b_[1][0],
                              cos * a_[1][1] + sin * b_[1][1]};
    });
    for (int i = 1; i <= n_steps_ && pass; i++) {
      UpdateModel();
      if (i % refresh_rate_ == 0) {
        filename = dir_ + std::to_string(i) + ".vtu";
        pass = OutputCurrentResult(filename);
      }
    }
    if (pass) {
      std::cout << "Complete calculation!" << std::endl;
    } else {
      std::cout << "Calculation failed!" << std::endl;
    }
  }

 private:
  bool OutputCurrentResult(std::string const& filename) {
    writer_.SetMesh(mesh_.get());
    return writer_.WriteToFile(filename);
  }
  void UpdateModel() {
    mesh_->ForEachWall([&](Wall& wall) {
      auto left_cell = wall.template GetSide<+1>();
      auto right_cell = wall.template GetSide<-1>();
      auto a_matrix = Matrix{wall.data.vectors[0][0], wall.data.vectors[0][1],
                             wall.data.vectors[1][0], wall.data.vectors[1][1]};
      auto riemann_ = Riemann(a_matrix);
      State u_l{0.0, 0.0}, u_r{0.0, 0.0};
      Flux f{0.0, 0.0};
      if (left_cell && right_cell) {
        u_l[0] = left_cell->data.scalars[0];
        u_l[1] = left_cell->data.scalars[1];
        u_r[0] = right_cell->data.scalars[0];
        u_r[1] = right_cell->data.scalars[1];
        f = riemann_.GetFluxOnTimeAxis(u_l, u_r);
      } else if (left_cell) {
        u_l[0] = left_cell->data.scalars[0];
        u_l[1] = left_cell->data.scalars[1];
        f = riemann_.GetFluxOnTimeAxis(u_l, u_l);
      } else {
        u_r[0] = right_cell->data.scalars[0];
        u_r[1] = right_cell->data.scalars[1];
        f = riemann_.GetFluxOnTimeAxis(u_r, u_r);
      }
      wall.data.scalars[0] = f[0];
      wall.data.scalars[1] = f[1];
    });
    mesh_->ForEachCell([&](Cell& cell) {
      State rhs = {0.0, 0.0};
      cell.ForEachWall([&](Wall& wall) {
        if (wall.template GetSide<+1>() == &cell) {
          rhs[0] -= wall.data.scalars[0] * wall.Measure();
          rhs[1] -= wall.data.scalars[1] * wall.Measure();
        } else {
          rhs[0] += wall.data.scalars[0] * wall.Measure();
          rhs[1] += wall.data.scalars[1] * wall.Measure();
        }
      });
      rhs[0] /= cell.Measure();
      rhs[1] /= cell.Measure();
      TimeStepping(&(cell.data.scalars[0]), rhs[0]);
      TimeStepping(&(cell.data.scalars[1]), rhs[1]);
      cell.data.scalars[2] = cell.data.scalars[0] - cell.data.scalars[1];
    });
  }
  void TimeStepping(double* u_curr , double du_dt) {
    *u_curr += du_dt * step_size_;
  }
  Matrix a_;
  Matrix b_;
  VtkReader reader_;
  VtkWriter writer_;
  std::unique_ptr<Mesh> mesh_;
  double duration_;
  int n_steps_;
  double step_size_;
  std::string dir_;
  int refresh_rate_;
};

}  // namespace model
}  // namespace mini

#endif  // MINI_MODEL_DOUBLE_WAVE_HPP_
