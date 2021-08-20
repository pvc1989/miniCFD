// Copyright 2021 PEI Weicheng and JIANG Yuyan
/**
 * This file defines parser of partition info txt.
 */
#ifndef MINI_MESH_CGNS_PARSER_HPP_
#define MINI_MESH_CGNS_PARSER_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>

#include "pcgnslib.h"

#include "mini/mesh/cgns/format.hpp"

namespace mini {
namespace mesh {
namespace cgns {

template <class Int = cgsize_t>
struct NodeInfo {
  NodeInfo() = default;
  NodeInfo(Int zi, Int ni) : zone_id(zi), node_id(ni) {}
  NodeInfo(NodeInfo const&) = default;
  NodeInfo& operator=(NodeInfo const&) = default;
  NodeInfo(NodeInfo&&) noexcept = default;
  NodeInfo& operator=(NodeInfo&&) noexcept = default;
  ~NodeInfo() noexcept = default;
  Int zone_id{0}, node_id{0};
};
template <class Int = int>
struct CellInfo {
  CellInfo() = default;
  CellInfo(Int zi, Int si, Int ci) : zone_id(zi), section_id(si), cell_id(ci) {}
  Int zone_id{0}, section_id{0}, cell_id{0};
};

template <typename Int = cgsize_t, typename Real = double>
struct NodeGroup {
  Int head_, size_;
  ShiftedVector<Int> metis_id_;
  ShiftedVector<Real> x_, y_, z_;

  NodeGroup(int head, int size)
      : head_(head), size_(size), metis_id_(size, head),
        x_(size, head), y_(size, head), z_(size, head) {
  }
  NodeGroup() = default;
  NodeGroup(NodeGroup const&) = default;
  NodeGroup(NodeGroup&&) noexcept = default;
  NodeGroup& operator=(NodeGroup const&) = default;
  NodeGroup& operator=(NodeGroup &&) noexcept = default;
  ~NodeGroup() noexcept = default;

  Int size() const {
    return size_;
  }
};

template <typename Int = cgsize_t, typename Real = double>
struct CellBase {
  Int metis_id;
};

template <typename Int = cgsize_t, typename Real = double>
class CellGroup {
  Int head_, tail_;
  std::vector<std::unique_ptr<CellBase<Int, Real>>> data_;
};

template <typename Int = cgsize_t, typename Real = double>
class Parser{
  static constexpr int kLineWidth = 30;

 public:
  Parser(std::string const& cgns_file, std::string const& prefix, int pid) {
    int fid;
    if (cgp_open(cgns_file.c_str(), CG_MODE_READ, &fid)) {
      cgp_error_exit();
    }
    auto filename = prefix + std::to_string(pid) + ".txt";
    std::ifstream istrm(filename);
    char line[kLineWidth];
    // node ranges
    while (istrm.getline(line, 30) && line[0]) {
      int zid, head, tail;
      std::sscanf(line, "%d %d %d", &zid, &head, &tail);
      nodes[zid] = NodeGroup<Int, Real>(head, tail - head);
      cgsize_t range_min = head, range_max = tail - 1;
      cgp_coord_read_data(fid, 1, zid, 1, &range_min, &range_max,
          nodes[zid].x_.data());
      cgp_coord_read_data(fid, 1, zid, 2, &range_min, &range_max,
          nodes[zid].y_.data());
      cgp_coord_read_data(fid, 1, zid, 3, &range_min, &range_max,
          nodes[zid].z_.data());
      cgp_field_read_data(fid, 1, zid, 1, 2, &range_min, &range_max,
          nodes[zid].metis_id_.data());
      for (int nid = head; nid < tail; ++nid) {
        auto mid = nodes[zid].metis_id_[nid];
        m_to_c_for_nodes[mid] = NodeInfo<Int>(zid, nid);
        std::cout << mid << ": " << zid << " " << nid << std::endl;
      }
      cgp_close(fid);
    }
    // adjacent nodes
    while (istrm.getline(line, 30) && line[0]) {
      int p, node;
      std::sscanf(line, "%d %d", &p, &node);
      part_adj_nodes[p].emplace(node);
    }
    // cell ranges
    while (istrm.getline(line, 30) && line[0]) {
      int zid, s, head, tail;
      std::sscanf(line, "%d %d %d %d", &zid, &s, &head, &tail);
      cells[zid][s] = {head, tail};
    }
    // inner adjacency
    while (istrm.getline(line, 30) && line[0]) {
      int i, j;
      std::sscanf(line, "%d %d", &i, &j);
      inner_adjs.emplace_back(i, j);
    }
    // interpart adjacency
    while (istrm.getline(line, 30) && line[0]) {
      int p, i, j;
      std::sscanf(line, "%d %d %d", &p, &i, &j);
      part_interpart_adjs[p].emplace_back(i, j);
    }
  }

 private:
  std::map<Int, NodeGroup<Int, Real>> nodes;
  std::unordered_map<Int, NodeInfo<Int>> m_to_c_for_nodes;
  std::map<Int, std::set<Int>> part_adj_nodes;
  std::map<Int, std::map<Int, std::pair<Int, Int>>> cells;
  std::vector<std::pair<Int, Int>> inner_adjs;
  std::map<Int, std::vector<std::pair<Int, Int>>> part_interpart_adjs;
  
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_PARSER_HPP_
