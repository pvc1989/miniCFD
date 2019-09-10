// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MODEL_SINGLE_WAVE_HPP_
#define MINI_MODEL_SINGLE_WAVE_HPP_

#include <memory>
#include <string>

#include "mini/riemann/linear.hpp"
#include "mini/mesh/data.hpp"
#include "mini/mesh/dim2.hpp"
#include "mini/mesh/vtk.hpp"

namespace mini {
namespace model {

template <class Mesh, class Riemann>
class SingleWave {
  using Boundary = typename Mesh::Boundary;
  using Domain = typename Mesh::Domain;
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
      return true;
    } else {
      return false;
    }
  }
  // Mutators:
  template <class Visitor>
  void SetInitialState(Visitor&& visitor) {
    mesh_->ForEachDomain(visitor);
  }
  template <class Visitor>
  void SetWallBoundary(Visitor&& visitor) {
    mesh_->ForEachBoundary(visitor);
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
    mesh_->ForEachBoundary([&](Boundary& boundary){
      double cos = (boundary.Tail()->Y() - boundary.Head()->Y()) /
                    boundary.Measure();
      double sin = (boundary.Head()->X() - boundary.Tail()->X()) /
                    boundary.Measure();
      double a = cos * a_ + sin * b_;
      boundary.data.scalars[1] = a;
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
    auto riemann_solver = [&](Boundary& boundary) {
      auto left_domain = boundary.template GetSide<+1>();
      auto right_domain = boundary.template GetSide<-1>();
      if (left_domain && right_domain) {
        auto riemann = Riemann(boundary.data.scalars[1]);
        State u_l = left_domain->data.scalars[0];
        State u_r = right_domain->data.scalars[0];
        Flux f = riemann.GetFluxOnTimeAxis(u_l, u_r);
        boundary.data.scalars[0] = f;
      } else if (left_domain) {
        boundary.data.scalars[0] = left_domain->data.scalars[0] *
                                      boundary.data.scalars[1];
      } else {
        boundary.data.scalars[0] = right_domain->data.scalars[0] *
                                      boundary.data.scalars[1];
      }
    };
    auto get_next_u = [&](Domain& domain) {
      double rhs = 0.0;
      domain.ForEachBoundary([&](Boundary& boundary) {
        if (boundary.template GetSide<+1>() == &domain) {
          rhs -= boundary.data.scalars[0] * boundary.Measure();
        } else {
          rhs += boundary.data.scalars[0] * boundary.Measure();
        }
      });
      rhs /= domain.Measure();
      TimeStepping(domain.data.scalars[0], rhs);
    };
    mesh_->ForEachBoundary(riemann_solver);
    mesh_->ForEachDomain(get_next_u);
  }
  void TimeStepping(double& u_curr , double du_dt) {
    u_curr += du_dt * step_size_;
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
};

}  // namespace model
}  // namespace mini

#endif  // MINI_MODEL_SINGLE_WAVE_HPP_
