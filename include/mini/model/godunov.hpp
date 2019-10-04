// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MODEL_GODUNOV_HPP_
#define MINI_MODEL_GODUNOV_HPP_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mini/mesh/data.hpp"
#include "mini/mesh/dim2.hpp"
#include "mini/mesh/vtk.hpp"

namespace mini {
namespace model {

template <class Mesh, class Riemann>
class Godunov {
  using Wall = typename Mesh::Wall;
  using Cell = typename Mesh::Cell;
  using State = typename Riemann::State;
  using Flux = typename Riemann::Flux;
  using Reader = mesh::VtkReader<Mesh>;
  using Writer = mesh::VtkWriter<Mesh>;

 public:
  explicit Godunov(std::string const& name) : model_name_(name) {}
  bool ReadMesh(std::string const& file_name) {
    reader_ = Reader();
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
    assert(CheckBoundarycondition());
    writer_ = Writer();
    // Write the frame of initial state:
    auto filename = dir_ + model_name_ + "." + std::to_string(0) + ".vtu";
    bool pass = WriteCurrentFrame(filename);
    assert(pass);
    // Write other steps:
    for (int i = 1; i <= n_steps_ && pass; i++) {
      UpdateModel();
      if (i % refresh_rate_ == 0) {
        filename = dir_ + model_name_ + "." +std::to_string(i) + ".vtu";
        pass = WriteCurrentFrame(filename);
      }
      std::printf("Progress: %d/%d\n", i, n_steps_);
    }
  }

 private:
  bool WriteCurrentFrame(std::string const& filename) {
    mesh_->ForEachCell([&](Cell& cell) {
      cell.data.Write();
    });
    writer_.SetMesh(mesh_.get());
    return writer_.WriteToFile(filename);
  }
  void Preprocess() {
    mesh_->ForEachWall([&](Wall& wall){
      auto length = wall.Measure();
      auto n1 = (wall.Tail()->Y() - wall.Head()->Y()) / length;
      auto n2 = (wall.Head()->X() - wall.Tail()->X()) / length;
      wall.data.riemann.Rotate(n1, n2);
      auto left_cell = wall.GetPositiveSide();
      auto right_cell = wall.GetNegativeSide();
      if (left_cell && right_cell) {
        inside_walls_.emplace(&wall);
      } else {
        boundary_walls_.emplace(&wall);
      }
    });
  }
  void UpdateModel() {
    mesh_->ForEachWall([&](Wall& wall) {
      auto left_cell = wall.GetPositiveSide();
      auto right_cell = wall.GetNegativeSide();
      auto const& riemann_ = &wall.data.riemann;
      if (left_cell && right_cell) {
        auto const& u_l = left_cell->data.state;
        auto const& u_r = right_cell->data.state;
        wall.data.flux = riemann_->GetFluxOnTimeAxis(u_l, u_r);
      } else if (left_cell) {
        auto const& u_l = left_cell->data.state;
        wall.data.flux = riemann_->GetFluxOnTimeAxis(u_l, u_l);
      } else {
        auto const& u_r = right_cell->data.state;
        wall.data.flux = riemann_->GetFluxOnTimeAxis(u_r, u_r);
      }
      wall.data.flux *= wall.Measure();
    });
    mesh_->ForEachCell([&](Cell& cell) {
      auto net_flux = Flux{};
      cell.ForEachWall([&](Wall& wall) {
        if (wall.GetPositiveSide() == &cell) {
          net_flux -= wall.data.flux;
        } else {
          net_flux += wall.data.flux;
        }
      });
      net_flux /= cell.Measure();
      TimeStepping(&(cell.data.state), &net_flux);
    });
  }
  void TimeStepping(State* u_curr , Flux* du_dt) {
    *du_dt *= step_size_;
    *u_curr += *du_dt;
  }
  bool CheckBoundarycondition() {
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
    auto in_l = a->GetPositiveSide();
    auto in_r = a->GetNegativeSide();
    auto out_l = b->GetPositiveSide();
    auto out_r = b->GetNegativeSide();
    if (in_l == nullptr) {
      if (out_l == nullptr) {
        a->SetPositiveSide(out_r);
        b->SetPositiveSide(in_r);
      } else {
        a->SetPositiveSide(out_l);
        b->SetNegativeSide(in_r);
      }
    } else {
      if (out_l == nullptr) {
        a->SetNegativeSide(out_r);
        b->SetPositiveSide(in_l);
      } else {
        a->SetNegativeSide(out_l);
        b->SetNegativeSide(in_l);
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
      auto const& riemann_ = wall->data.riemann;
      auto const& u_l = wall->GetPositiveSide()->data.state;
      auto const& u_r = wall->GetNegativeSide()->data.state;
      wall.data.flux = riemann_->GetFluxOnTimeAxis(u_l, u_r);
    }
  }
  void CalculateFreeBoundary() {
    for (auto& name : free_boundaries_) {
      for (auto& wall : boundaries_[name]) {
        auto const& riemann_ = wall->data.riemann;
        if (wall->GetPositiveSide()) {
          auto const& u = wall->GetPositiveSide()->data.state;
          wall.data.flux = riemann_->GetFluxOnTimeAxis(u, u);
        } else {
          auto const& u = wall->GetNegativeSide()->data.state;
          wall.data.flux = riemann_->GetFluxOnTimeAxis(u, u);
        }
      }
    }
  }
  void CalculateSolidBoundary() {
    for (auto& name : free_boundaries_) {
      for (auto& wall : boundaries_[name]) {
        auto const& riemann_ = wall->data.riemann;
        if (wall->GetPositiveSide()) {
          auto const& u = wall->GetPositiveSide()->data.state;
          wall.data.flux = riemann_->GetFluxOnTimeAxis(u, -u);
        } else {
          auto const& u = wall->GetNegativeSide()->data.state;
          wall.data.flux = riemann_->GetFluxOnTimeAxis(-u, u);
        }
      }
    }
  }

 private:
  std::string model_name_;
  Reader reader_;
  Writer writer_;
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

#endif  // MINI_MODEL_GODUNOV_HPP_
