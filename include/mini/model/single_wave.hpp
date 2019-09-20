// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MODEL_SINGLE_WAVE_HPP_
#define MINI_MODEL_SINGLE_WAVE_HPP_

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mini/riemann/linear.hpp"
#include "mini/mesh/data.hpp"
#include "mini/mesh/dim2.hpp"
#include "mini/mesh/vtk.hpp"

namespace mini {
namespace model {

template <class Mesh, class Riemann>
class SingleWave {
  using Wall = typename Mesh::Wall;
  using Cell = typename Mesh::Cell;
  using State = typename Riemann::State;
  using Flux = typename Riemann::Flux;
  using VtkReader = typename mesh::VtkReader<Mesh>;
  using VtkWriter = typename mesh::VtkWriter<Mesh>;

 public:
  explicit SingleWave(double a, double b) {
    a_ = a;
    b_ = b;
  }
  bool ReadMesh(std::string const& file_name) {
    reader_ = VtkReader();
    if (reader_.ReadFromFile(file_name)) {
      mesh_ = reader_.GetMesh();
      Preprocess();
      return true;
    } else {
      return false;
    }
  }
  // Mutators:
  template <class Visitor>
  void SetBoundaryName(std::string const& name, Visitor&& visitor) {
    boundaries_.emplace(name, std::vector<Wall*>());
    auto& part = boundaries_[name];
    for (auto& wall : boundary_walls_) {
      if (visitor(*wall)) {
        part.emplace_back(wall);
      }
    }
  }
  void SetInletBoundary(std::string const& name) {
    inlet_boundaries_.emplace(name);
  }
  void SetOutletBoundary(std::string const& name) {
    outlet_boundaries_.emplace(name);
  }
  void SetPeriodicBoundary(std::string const& name_a,
                           std::string const& name_b) {
    auto& part_a = boundaries_[name_a];
    auto& part_b = boundaries_[name_b];
    assert(part_a.size() == part_b.size());
    periodic_boundaries_.emplace(name_a, name_b);
    auto cmp = [](Wall* a, Wall* b) {
      auto point_a = a->Center();
      auto point_b = b->Center();
      if (point_a.Y() != point_b.Y()) {
        return point_a.Y() < point_b.Y();
      } else {
        return point_a.X() < point_b.X();
      }
    };
    std::sort(part_a.begin(), part_a.end(), cmp);
    std::sort(part_b.begin(), part_b.end(), cmp);
    for (int i = 0; i < part_a.size(); i++) {
      SewEndsOfWalls(part_a[i], part_b[i]);
    }
  }
  void SetFreeBoundary(std::string const& name) {
    free_boundaries_.emplace(name);
  }
  void SetSolidBoundary(std::string const& name) {
    solid_boundaries_.emplace(name);
  }
  // void SetSolidWallCondition() {}
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
  void SetOutputDir(std::string const& dir) {
    dir_ = dir;
  }
  // Major computation:
  void Calculate() {
    assert(CheckBoundarycondition());
    writer_ = VtkWriter();
    auto filename = dir_ + std::to_string(0) + ".vtu";
    bool pass = OutputCurrentResult(filename);
    assert(pass);
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
  void Preprocess() {
    mesh_->ForEachWall([&](Wall& wall){
      auto length = wall.Measure();
      double cos = (wall.Tail()->Y() - wall.Head()->Y()) / length;
      double sin = (wall.Head()->X() - wall.Tail()->X()) / length;
      wall.data.scalars[1] = cos * a_ + sin * b_;
      auto left_cell = wall.template GetSide<+1>();
      auto right_cell = wall.template GetSide<-1>();
      if (left_cell && right_cell) {
        inside_walls_.emplace(&wall);
      } else {
        boundary_walls_.emplace(&wall);
      }
    });
  }
  bool CheckBoundarycondition() {
    for (auto& [left, right] : periodic_boundaries_) {
      std::cout << left << " " << right << std::endl;
    }
    int n = 0;
    for (auto& [name, part] : boundaries_) {
      n += part.size();
    }
    if (n == boundary_walls_.size()) {
      boundary_walls_.clear();
    } else {
      std::cerr << "Lost boundaries!" << std::endl;
      return false;
    }
    return true;
  }
  void SewEndsOfWalls(Wall* a, Wall* b) {
    auto in_l = a->template GetSide<+1>();
    auto in_r = a->template GetSide<-1>();
    auto out_l = b->template GetSide<+1>();
    auto out_r = b->template GetSide<-1>();
    if (in_l == nullptr) {
      if (out_l == nullptr) {
        a->template SetSide<+1>(out_r);
        b->template SetSide<+1>(in_r);
      } else {
        a->template SetSide<+1>(out_l);
        b->template SetSide<-1>(in_r);
      }
    } else {
      if (out_l == nullptr) {
        a->template SetSide<-1>(out_r);
        b->template SetSide<+1>(in_l);
      } else {
        a->template SetSide<-1>(out_l);
        b->template SetSide<-1>(in_l);
      }
    }
    inside_walls_.emplace(a);
    inside_walls_.emplace(b);
  }
  void CalculateEachWall() {
    CalculateInsideWalls();
    CalculateFreeBoundary();
    CalculateSolidBoundary();
  }
  void CalculateInsideWalls() {
    for (auto& wall : inside_walls_) {
      auto riemann_ = Riemann(wall->data.scalars[1]);
      auto u_l = wall->template GetSide<+1>()->data.scalars[0];
      auto u_r = wall->template GetSide<-1>()->data.scalars[0];
      wall->data.scalars[0] = riemann_.GetFluxOnTimeAxis(u_l, u_r);
    }
  }
  void CalculateFreeBoundary() {
    for (auto& name : free_boundaries_) {
      for (auto& wall : boundaries_[name]) {
        auto riemann_ = Riemann(wall->data.scalars[1]);
        auto u = State(0.0);
        if (wall->template GetSide<+1>()) {
          u = wall->template GetSide<+1>()->data.scalars[0];
        } else {
          u = wall->template GetSide<-1>()->data.scalars[0];
        }
        auto f = riemann_.GetFluxOnTimeAxis(u, u);
        wall->data.scalars[0] = f;
      }
    }
  }
  void CalculateSolidBoundary() {
    for (auto& name : free_boundaries_) {
      for (auto& wall : boundaries_[name]) {
        auto riemann_ = Riemann(wall->data.scalars[1]);
        auto u = State(0.0);
        auto f = Flux(0.0);
        if (wall->template GetSide<+1>()) {
          u = wall->template GetSide<+1>()->data.scalars[0];
          f = riemann_.GetFluxOnTimeAxis(u, -u);
        } else {
          u = wall->template GetSide<-1>()->data.scalars[0];
          f = riemann_.GetFluxOnTimeAxis(-u, u);
        }
        wall->data.scalars[0] = f;
      }
    }
  }
  void UpdateModel() {
    CalculateEachWall();
    mesh_->ForEachCell([&](Cell& cell) {
      double rhs = 0.0;
      cell.ForEachWall([&](Wall& wall) {
        if (wall.template GetSide<+1>() == &cell) {
          rhs -= wall.data.scalars[0] * wall.Measure();
        } else {
          rhs += wall.data.scalars[0] * wall.Measure();
        }
      });
      rhs /= cell.Measure();
      TimeStepping(&(cell.data.scalars[0]), rhs);
    });
  }
  void TimeStepping(double* u_curr , double du_dt) {
    *u_curr += du_dt * step_size_;
  }
  double a_;
  double b_;
  VtkReader reader_;
  VtkWriter writer_;
  std::unique_ptr<Mesh> mesh_;
  double duration_;
  int n_steps_;
  double step_size_;
  std::string dir_;
  int refresh_rate_;
  std::set<Wall*> inside_walls_;
  std::set<Wall*> boundary_walls_;
  std::unordered_map<std::string, std::vector<Wall*>> boundaries_;
  std::set<std::string> inlet_boundaries_;
  std::set<std::string> outlet_boundaries_;
  std::set<std::pair<std::string, std::string>> periodic_boundaries_;
  std::set<std::string> free_boundaries_;
  std::set<std::string> solid_boundaries_;
};
}  // namespace model
}  // namespace mini

#endif  // MINI_MODEL_SINGLE_WAVE_HPP_
