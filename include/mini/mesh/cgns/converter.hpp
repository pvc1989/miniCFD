// Copyright 2020 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_CGNS_CONVERTER_HPP_
#define MINI_MESH_CGNS_CONVERTER_HPP_

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cgnslib.h"

#include "mini/mesh/cgns/tree.hpp"

namespace mini {
namespace mesh {
namespace cgns {

struct NodeInfo {
  int zone_id{0};
  int node_id{0};
};
struct ElementInfo {
  int zone_id{0};
  int section_id{0};
  int element_id{0};
};

struct CompressedSparseRowMatrix {
  std::vector<int> pointer;
  std::vector<int> index;
};

struct MetisMesh {

};

struct Converter {
  using Meshtype = Tree<double>;
  Converter() = default;
  MetisMesh ConvertToMetisMesh(const Meshtype* mesh);
  std::vector<NodeInfo> node_id_to_z_n;
  std::vector<std::vector<int>> z_n_to_node_id;
  std::vector<ElementInfo> elem_id_to_z_s_e;
};

MetisMesh Converter::ConvertToMetisMesh(const Converter::Meshtype* mesh) {
  assert(mesh->CountBases() == 1);
  auto& base = mesh->GetBase(1);
  auto cell_dim = base.GetCellDim();
  auto nzones = base.CountZones();
  z_n_to_node_id = std::vector<std::vector<int>>(nzones+1);
  int nelements{0};
  int nnodes{0};
  for (int zone_id = 1; zone_id <= nzones; ++zone_id) {
    auto& zone = base.GetZone(zone_id);
    auto n_nodes = zone.GetVertexSize();
    auto n_section = zone.CountSections();
  }
}

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_CONVERTER_HPP_