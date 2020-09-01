// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MODEL_GODUNOV_HPP_
#define MINI_MODEL_GODUNOV_HPP_

#include <cstdio>
#include <memory>
#include <set>
#include <string>

#include "mini/mesh/vtk/reader.hpp"
#include "mini/mesh/vtk/writer.hpp"
#include "mini/model/boundary.hpp"

namespace mini {
namespace model {

template <class Mesh, class Riemann>
class Godunov {
  using WallType = typename Mesh::WallType;
  using CellType = typename Mesh::CellType;
  using State = typename Riemann::State;
  using FluxType = typename Riemann::FluxType;
  using Reader = mesh::vtk::Reader<Mesh>;
  using Writer = mesh::vtk::Writer<Mesh>;

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
    wall_manager_.SetBoundaryName(name, visitor);
  }
  void SetInletBoundary(std::string const& name, State& inlet) {
    wall_manager_.SetInletBoundary(name);
    inlet_ = inlet;
  }
  void SetFreeBoundary(std::string const& name) {
    wall_manager_.SetFreeBoundary(name);
  }
  void SetPeriodicBoundary(std::string const& name_a,
                           std::string const& name_b) {
    wall_manager_.SetPeriodicBoundary(name_a, name_b);
  }
  void SetSolidBoundary(std::string const& name) {
    wall_manager_.SetSolidBoundary(name);
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
    wall_manager_.ClearBoundaryCondition();
    writer_ = Writer();
    // Write the frame of initial state:
    auto filename = dir_ + model_name_ + "." + std::to_string(0) + ".vtu";
    bool pass = WriteCurrentFrame(filename);
    assert(pass);
    // Write other steps:
    for (int i = 1; i <= n_steps_ && pass; i++) {
      UpdateEachWall();
      UpdateEachCell();
      if (i % refresh_rate_ == 0) {
        filename = dir_ + model_name_ + "." +std::to_string(i) + ".vtu";
        pass = WriteCurrentFrame(filename);
      }
      std::printf("Progress: %d/%d\n", i, n_steps_);
    }
  }

 private:
  bool WriteCurrentFrame(std::string const& filename) {
    mesh_->ForEachCell([&](CellType& cell) {
      cell.data.Write();
    });
    writer_.SetMesh(mesh_.get());
    return writer_.WriteToFile(filename);
  }
  void Preprocess() {
    mesh_->ForEachWall([&](WallType& wall){
      auto length = wall.Measure();
      auto n1 = (wall.Tail().Y() - wall.Head().Y()) / length;
      auto n2 = (wall.Head().X() - wall.Tail().X()) / length;
      wall.data.riemann.Rotate(n1, n2);
      auto left_cell = wall.GetPositiveSide();
      auto right_cell = wall.GetNegativeSide();
      if (left_cell && right_cell) {
        wall_manager_.AddInteriorWall(&wall);
      } else {
        wall_manager_.AddBoundaryWall(&wall);
      }
    });
  }
  void UpdateEachWall() {
    wall_manager_.ForEachInteriorWall([](WallType* wall){
      auto& riemann_ = wall->data.riemann;
      auto const& u_l = wall->GetPositiveSide()->data.state;
      auto const& u_r = wall->GetNegativeSide()->data.state;
      wall->data.flux = riemann_.GetFluxOnTimeAxis(u_l, u_r);
      wall->data.flux *= wall->Measure();
    });
    wall_manager_.ForEachInletWall([&](WallType* wall){
      auto& riemann_ = wall->data.riemann;
      auto const& u_l = wall->GetPositiveSide()->data.state;
      auto const& u_r = wall->GetNegativeSide()->data.state;
      auto left_cell = wall->GetPositiveSide();
      auto right_cell = wall->GetNegativeSide();
      if (left_cell) {
        auto& u_l = left_cell->data.state;
        wall->data.flux = riemann_.GetFluxOnTimeAxis(u_l, inlet_);
      } else {
        auto& u_r = right_cell->data.state; 
        wall->data.flux = riemann_.GetFluxOnTimeAxis(inlet_, u_r);
      }
      wall->data.flux *= wall->Measure();
    });
    wall_manager_.ForEachPeriodicWall([](WallType* wall){
      auto& riemann_ = wall->data.riemann;
      auto const& u_l = wall->GetPositiveSide()->data.state;
      auto const& u_r = wall->GetNegativeSide()->data.state;
      wall->data.flux = riemann_.GetFluxOnTimeAxis(u_l, u_r);
      wall->data.flux *= wall->Measure();
    });
    wall_manager_.ForEachFreeWall([](WallType* wall){
      auto& riemann_ = wall->data.riemann;
      auto left_cell = wall->GetPositiveSide();
      auto right_cell = wall->GetNegativeSide();
      if (left_cell) {
        auto const& u = wall->GetPositiveSide()->data.state;
        wall->data.flux = riemann_.GetFluxOnFreeWall(u);
      } else {
        auto const& u = wall->GetNegativeSide()->data.state;
        wall->data.flux = riemann_.GetFluxOnFreeWall(u);
      }
      wall->data.flux *= wall->Measure();
    });
    wall_manager_.ForEachSolidWall([](WallType* wall){
      auto& riemann_ = wall->data.riemann;
      auto left_cell = wall->GetPositiveSide();
      auto right_cell = wall->GetNegativeSide();
      if (left_cell) {
        auto const& u = wall->GetPositiveSide()->data.state;
        wall->data.flux = riemann_.GetFluxOnSolidWall(u);
      } else {
        auto const& u = wall->GetNegativeSide()->data.state;
        wall->data.flux = riemann_.GetFluxOnSolidWall(u);
      }
      wall->data.flux *= wall->Measure();
    });
  }
  void UpdateEachCell() {
    mesh_->ForEachCell([&](CellType& cell) {
      auto net_flux = FluxType{};
      cell.ForEachWall([&](WallType& wall) {
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
  void TimeStepping(State* u_curr , FluxType* du_dt) {
    *du_dt *= step_size_;
    *u_curr += *du_dt;
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
  std::set<WallType*> inside_wall_;
  Manager<Mesh> wall_manager_;
  State inlet_;
};

}  // namespace model
}  // namespace mini

#endif  // MINI_MODEL_GODUNOV_HPP_
