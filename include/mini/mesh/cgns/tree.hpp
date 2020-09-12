// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_CGNS_TREE_HPP_
#define MINI_MESH_CGNS_TREE_HPP_

#include <memory>
#include <string>

namespace mini {
namespace mesh {
namespace cgns {

class Base {
 public:
  std::string const& GetName() const;
  int GetId() const;
  int GetCellDim() const;
  int GetPhysDim() const;
};

template <class Real>
class Tree {
 public:
  Base const& GetBase(int base_id) const;
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_TREE_HPP_
