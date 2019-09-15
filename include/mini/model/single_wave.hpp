// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MODEL_SINGLE_WAVE_HPP_
#define MINI_MODEL_SINGLE_WAVE_HPP_

#include <algorithm>
#include <memory>
#include <set>
#include <string>
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
  void SetInletCondition(Visitor&& visitor) {
    for (auto& wall : boundaries_) {
      if (visitor(*wall)) {
        inlet_boundaries_.emplace_back(wall);
      }
    }
  }
  template <class Visitor>
  void SetOutletCondition(Visitor&& visitor) {
    for (auto& wall : boundaries_) {
      if (visitor(*wall)) {
        outlet_boundaries_.emplace_back(wall);
      }
    }
  }
  void SetPeriodicCondition() {
    assert(inlet_boundaries_.size() == outlet_boundaries_.size());
    std::sort(outlet_boundaries_.begin(), outlet_boundaries_.end(),
              [](Wall* a, Wall* b) {return a->Center().Y() < b->Center().Y();});
    std::sort(inlet_boundaries_.begin(), inlet_boundaries_.end(),
              [](Wall* a, Wall* b) {return a->Center().Y() < b->Center().Y();});
    for (int i = 0; i < inlet_boundaries_.size(); i++) {
      auto in_l = inlet_boundaries_[i]->template GetSide<+1>();
      auto in_r = inlet_boundaries_[i]->template GetSide<-1>();
      auto out_l = outlet_boundaries_[i]->template GetSide<+1>();
      auto out_r = outlet_boundaries_[i]->template GetSide<-1>();
      if (in_l == nullptr) {
        if (out_l == nullptr) {
          inlet_boundaries_[i]->template SetSide<+1>(out_r);
          outlet_boundaries_[i]->template SetSide<+1>(in_r);

        } else {
          inlet_boundaries_[i]->template SetSide<+1>(out_l);
          outlet_boundaries_[i]->template SetSide<-1>(in_r);
        }
      } else {
        if (out_l == nullptr) {
          inlet_boundaries_[i]->template SetSide<-1>(out_r);
          outlet_boundaries_[i]->template SetSide<+1>(in_l);

        } else {
          inlet_boundaries_[i]->template SetSide<-1>(out_l);
          outlet_boundaries_[i]->template SetSide<-1>(in_l);
        }
      }
    }
    is_periodic_ = true;
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
  void SetOutputDir(std::string dir) {
    dir_ = dir;
  }
  // Major computation:
  void Calculate() {
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
      double cos = (wall.Tail()->Y() - wall.Head()->Y()) /
                    wall.Measure();
      double sin = (wall.Head()->X() - wall.Tail()->X()) /
                    wall.Measure();
      double a = cos * a_ + sin * b_;
      wall.data.scalars[1] = a;
      auto left_cell = wall.template GetSide<+1>();
      auto right_cell = wall.template GetSide<-1>();
      if (left_cell && right_cell ) {
        inside_walls_.insert(&wall);
      } else {
        boundaries_.insert(&wall);
      }
    });
  }
  void CalculateEachWall() {
    assert(is_periodic_);
    mesh_->ForEachWall([&](Wall& wall) {
      auto left_cell = wall.template GetSide<+1>();
      auto right_cell = wall.template GetSide<-1>();
      auto riemann_ = Riemann(wall.data.scalars[1]);
      State u_l{0.0}, u_r{0.0};
      Flux f{0.0};
      if (left_cell && right_cell) {
        u_l = left_cell->data.scalars[0];
        u_r = right_cell->data.scalars[0];
        f = riemann_.GetFluxOnTimeAxis(u_l, u_r);
      } else if (left_cell) {
        u_l = left_cell->data.scalars[0];
        f = riemann_.GetFluxOnTimeAxis(u_l, u_l);
      } else if (right_cell) {
        u_r = right_cell->data.scalars[0];
        f = riemann_.GetFluxOnTimeAxis(u_r, u_r);
      }
      wall.data.scalars[0] = f;
    });
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
  bool is_periodic_;
  std::set<Wall*> inside_walls_;
  std::set<Wall*> boundaries_;
  std::vector<Wall*> inlet_boundaries_;
  std::vector<Wall*> outlet_boundaries_;
};
}  // namespace model
}  // namespace mini

#endif  // MINI_MODEL_SINGLE_WAVE_HPP_
