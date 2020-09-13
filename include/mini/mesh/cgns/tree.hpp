// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_CGNS_TREE_HPP_
#define MINI_MESH_CGNS_TREE_HPP_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cgnslib.h"

namespace mini {
namespace mesh {
namespace cgns {

template <class Real>
class Base {
 public:
  Base() = default;
  Base(char* name, int id, int cell_dim, int phys_dim)
    : name_(name), id_(id), cell_dim_(cell_dim), phys_dim_(phys_dim) {}
  int GetId() const {
    return id_;
  }
  int GetCellDim() const {
    return cell_dim_;
  }
  int GetPhysDim() const {
    return phys_dim_;
  }
  std::string GetName() const {
    return name_;
  }

 private: 
  int id_;
  int cell_dim_;
  int phys_dim_;
  std::string name_;
};

template <class Real>
class Tree {
 public:
  // Types:
  using Base = Base<Real>;
  Tree() = default;
  bool OpenFile(const std::string& file_name) {
    name_ = file_name;
    if (cg_open(name_.c_str(), CG_MODE_READ, &id_)) {
      cg_error_exit();
      return false;
    }
    cg_nbases(id_, &n_bases_);
    for (int base_id = 1; base_id <= n_bases_; ++base_id) {
      char base_name[33];
      int cell_dim{-1}, phys_dim{-1};
      cg_base_read(id_, base_id, base_name, &cell_dim, &phys_dim);
      // bases_[base_id] = std::make_unique<Base>(base_name, base_id,
      //                                          cell_dim, phys_dim);
      bases_.emplace(std::make_pair(base_id,
                                    std::make_unique<Base>(base_name, base_id,
                                                           cell_dim, phys_dim)));
    }
    cg_close(id_);
    return true;
  }
  int GetId() const {
    return id_;
  }
  std::string GetName() const {
    return name_;
  }
  int CountBases() const {
    return n_bases_;
  }
  Base& GetBase(int id) {
    return *(bases_.at(id).get());
  }

 private:
  int id_{-1};
  int n_bases_{0};
  std::string name_;
  std::map<int, std::unique_ptr<Base>> bases_;
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_TREE_HPP_
