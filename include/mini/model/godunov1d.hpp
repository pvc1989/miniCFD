// Copyright 2019 PEI Weicheng and YANG Minghao

#ifndef MINI_MODEL_GODUNOV1D_HPP_
#define MINI_MODEL_GODUNOV1D_HPP_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mini/dataset/vtk.hpp"

namespace mini {
namespace model {

template <class Mesh>
class Manager {
 public:
  // Types:
  using Node = typename Mesh::Node;
  using Part = std::vector<Node*>;
  // Mutators:
  void AddInteriorNode(Node* node) {
    interior_nodes_.push_back(node);
  }
  void AddBoundaryNode(Node* node) {
    boundary_nodes_.push_back(node);
  }
  template <class Visitor>
  void SetBoundaryName(std::string const& name, Visitor&& visitor) {
    for (auto& node : boundary_nodes_) {
      if (visitor(*node)) {
        name_to_node_[name] = node;
      }
    }
  }
  void SetPeriodicBoundary(std::string const& head, std::string const& tail) {
    SewMatchingNodes(name_to_node_[head], name_to_node_[tail]);
  }
  void SetFreeBoundary(std::string const& name) {
    free_nodes_.push_back(name_to_node_[name]);
  }
  void SetSolidBoundary(std::string const& name) {
    solid_nodes_.push_back(name_to_node_[name]);
  }
  void ClearBoundaryCondition() {
    if (CheckBoundaryConditions()) {
      boundary_nodes_.clear();
    } else {
      throw std::length_error("Some `Node`s do not have BC info.");
    }
  }
  // Iterators:
  template<class Visitor>
  void ForEachInteriorNode(Visitor&& visit) {
    for (auto& node : interior_nodes_) {
      visit(node);
    }
  }
  template<class Visitor>
  void ForEachFreeNode(Visitor&& visit) {
    for (auto& node : free_nodes_) {
      visit(node);
    }
  }
  template<class Visitor>
  void ForEachSolidNode(Visitor&& visit) {
    for (auto& node : solid_nodes_) {
      visit(node);
    }
  }

 private:
  // Data members:
  Part interior_nodes_;
  Part boundary_nodes_;
  Part free_nodes_;
  Part solid_nodes_;
  std::unordered_map<std::string, Node*> name_to_node_;
  // Implement details:
  void SewMatchingNodes(Node* a, Node* b) {
    auto left___in = a->GetPositiveSide();
    auto right__in = a->GetNegativeSide();
    auto left__out = b->GetPositiveSide();
    auto right_out = b->GetNegativeSide();
    if (left___in == nullptr) {
      if (left__out == nullptr) {
        a->SetPositiveSide(right_out);
        b->SetPositiveSide(right__in);
      } else {
        a->SetPositiveSide(left__out);
        b->SetNegativeSide(right__in);
      }
    } else {
      if (left__out == nullptr) {
        a->SetNegativeSide(right_out);
        b->SetPositiveSide(left___in);
      } else {
        a->SetNegativeSide(left__out);
        b->SetNegativeSide(left___in);
      }
    }
    interior_nodes_.push_back(a);
    interior_nodes_.push_back(b);
  }
  bool CheckBoundaryConditions() {
    return name_to_node_.size() == boundary_nodes_.size();
  }
};

template <class Mesh, class Riemann>
class Godunov {
  using Node = typename Mesh::Node;
  using Cell = typename Mesh::Cell;
  using Conservative = typename Riemann::Conservative;
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
    node_manager_.SetBoundaryName(name, visitor);
  }
  void SetFreeBoundary(std::string const& name) {
    node_manager_.SetFreeBoundary(name);
  }
  void SetSolidBoundary(std::string const& name) {
    node_manager_.SetSolidBoundary(name);
  }
  void SetPeriodicBoundary(std::string const& name_a,
                           std::string const& name_b) {
    node_manager_.SetPeriodicBoundary(name_a, name_b);
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
    node_manager_.ClearBoundaryCondition();
    writer_ = Writer();
    // Write the frame of initial state:
    auto filename = dir_ + model_name_ + "." + std::to_string(0) + ".vtu";
    bool pass = WriteCurrentFrame(filename);
    assert(pass);
    // Write other steps:
    for (int i = 1; i <= n_steps_ && pass; i++) {
      UpdateEachNode();
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
    mesh_->ForEachCell([&](Cell& cell) {
      cell.data.Write();
    });
    writer_.SetMesh(mesh_.get());
    return writer_.WriteToFile(filename);
  }
  void Preprocess() {
    mesh_->ForEachNode([&](Node& node){
      auto left_cell = node.GetPositiveSide();
      auto right_cell = node.GetNegativeSide();
      if (left_cell && right_cell) {
        node_manager_.AddInteriorNode(&node);
      } else {
        node_manager_.AddBoundaryNode(&node);
      }
    });
  }
  void UpdateEachNode() {
    node_manager_.ForEachInteriorNode([this](Node* node){
      auto const& u_l = node->GetPositiveSide()->data.state;
      auto const& u_r = node->GetNegativeSide()->data.state;
      node->data.flux = this->riemann_.GetFluxOnTimeAxis(u_l, u_r);
    });
    node_manager_.ForEachFreeNode([this](Node* node){
      auto left_cell = node->GetPositiveSide();
      auto right_cell = node->GetNegativeSide();
      if (left_cell) {
        auto const& u = node->GetPositiveSide()->data.state;
        node->data.flux = this->riemann_.GetFluxOnFreeWall(u);
      } else {
        auto const& u = node->GetNegativeSide()->data.state;
        node->data.flux = this->riemann_.GetFluxOnFreeWall(u);
      }
    });
    node_manager_.ForEachSolidNode([this](Node* node){
      auto left_cell = node->GetPositiveSide();
      auto right_cell = node->GetNegativeSide();
      if (left_cell) {
        auto const& u = node->GetPositiveSide()->data.state;
        node->data.flux = this->riemann_.GetFluxOnSolidWall(u);
      } else {
        auto const& u = node->GetNegativeSide()->data.state;
        node->data.flux = this->riemann_.GetFluxOnSolidWall(u);
      }
    });
  }
  void UpdateEachCell() {
    mesh_->ForEachCell([&](Cell& cell) {
      auto net_flux = Flux{};
      cell.ForEachNode([&](Node& node) {
        if (node.GetPositiveSide() == &cell) {
          net_flux -= node.data.flux;
        } else {
          net_flux += node.data.flux;
        }
      });
      net_flux /= cell.Measure();
      TimeStepping(&(cell.data.state), &net_flux);
    });
  }
  void TimeStepping(Conservative* u_curr , Flux* du_dt) {
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
  std::set<Node*> inside_node_;
  Manager<Mesh> node_manager_;
  Conservative inlet_;
  Riemann riemann_;
};

}  // namespace model
}  // namespace mini

#endif  // MINI_MODEL_GODUNOV1D_HPP_
