// Copyright 2021 PEI Weicheng and JIANG Yuyan
/**
 * This file defines parser of partition info txt.
 */
#ifndef MINI_MESH_CGNS_PARSER_HPP_
#define MINI_MESH_CGNS_PARSER_HPP_

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "pcgnslib.h"
#include "Eigen/Dense"

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
  bool has(int nid) const {
    return head_ <= nid && nid < size_ + head_;
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
    rank_ = pid;
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
      local_nodes_[zid] = NodeGroup<Int, Real>(head, tail - head);
      cgsize_t range_min[] = { head };
      cgsize_t range_max[] = { tail - 1 };
      auto& x = local_nodes_[zid].x_;
      cgp_coord_read_data(fid, 1, zid, 1, range_min, range_max, x.data());
      auto& y = local_nodes_[zid].y_;
      cgp_coord_read_data(fid, 1, zid, 2, range_min, range_max, y.data());
      auto& z = local_nodes_[zid].z_;
      cgp_coord_read_data(fid, 1, zid, 3, range_min, range_max, z.data());
      auto& metis_id = local_nodes_[zid].metis_id_;
      cgsize_t mem_dimensions[] = { tail - head };
      cgsize_t mem_range_min[] = { 1 };
      cgsize_t mem_range_max[] = { mem_dimensions[0] };
      cgp_field_general_read_data(fid, 1, zid, 1, 2, range_min, range_max,
          sizeof(Int) == 8 ? CGNS_ENUMV(LongInteger) : CGNS_ENUMV(Integer),
          1, mem_dimensions, mem_range_min, mem_range_max, metis_id.data());
      for (int nid = head; nid < tail; ++nid) {
        auto mid = metis_id[nid];
        nodes_m_to_c_[mid] = NodeInfo<Int>(zid, nid);
      }
    }
    cgp_close(fid);
    std::map<Int, std::vector<Int>> send_nodes, recv_nodes;
    // send nodes info
    while (istrm.getline(line, 30) && line[0]) {
      int p, node;
      std::sscanf(line, "%d %d", &p, &node);
      send_nodes[p].emplace_back(node);
    }
    std::vector<MPI_Request> requests;
    std::vector<std::vector<Real>> send_bufs;
    for (auto& [target, nodes] : send_nodes) {
      auto& buf = send_bufs.emplace_back();
      for (auto metis_id : nodes) {
        auto& info = nodes_m_to_c_[metis_id];
        auto const& coord = GetCoord(info.zone_id, info.node_id);
        buf.emplace_back(coord[0]);
        buf.emplace_back(coord[1]);
        buf.emplace_back(coord[2]);
      }
      assert(std::is_sorted(nodes.begin(), nodes.end()));
      int count = 3 * nodes.size();
      MPI_Datatype datatype = MPI_DOUBLE;
      int tag = target;
      auto& req = requests.emplace_back();
      MPI_Isend(buf.data(), count, datatype, target, tag, MPI_COMM_WORLD, &req);
    }
    // recv nodes info
    while (istrm.getline(line, 30) && line[0]) {
      int p, node;
      std::sscanf(line, "%d %d", &p, &node);
      recv_nodes[p].emplace_back(node);
    }
    std::vector<std::vector<Real>> recv_bufs;
    for (auto& [source, nodes] : recv_nodes) {
      assert(std::is_sorted(nodes.begin(), nodes.end()));
      int count = 3 * nodes.size();
      auto& buf = recv_bufs.emplace_back(std::vector<Real>(count));
      MPI_Datatype datatype = MPI_DOUBLE;
      int tag = rank_;
      auto& req = requests.emplace_back();
      MPI_Irecv(buf.data(), count, datatype, source, tag, MPI_COMM_WORLD, &req);
    }
    std::vector<MPI_Status> statuses(requests.size());
    MPI_Waitall(requests.size(), requests.data(), statuses.data());
    // copy node coordinates from buffer to member
    int i_source = 0;
    for (auto& [source, nodes] : recv_nodes) {
      double* xyz = recv_bufs[i_source++].data();
      for (auto metis_id : nodes) {
        auto& info = nodes_m_to_c_[metis_id];
        adj_nodes_[info.zone_id][info.node_id] = { xyz[0], xyz[1] , xyz[2] };
        xyz += 3;
      }
    }
    // cell ranges
    while (istrm.getline(line, 30) && line[0]) {
      int zid, s, head, tail;
      std::sscanf(line, "%d %d %d %d", &zid, &s, &head, &tail);
      local_cells_[zid][s] = {head, tail};
    }
    // inner adjacency
    while (istrm.getline(line, 30) && line[0]) {
      int i, j;
      std::sscanf(line, "%d %d", &i, &j);
      inner_adjs_.emplace_back(i, j);
    }
    // interpart adjacency
    while (istrm.getline(line, 30) && line[0]) {
      int p, i, j;
      std::sscanf(line, "%d %d %d", &p, &i, &j);
      part_to_adjs_[p].emplace_back(i, j);
    }
  }

 private:
  using Mat3x1 = Eigen::Matrix<Real, 3, 1>;
  std::map<Int, NodeGroup<Int, Real>> local_nodes_;
  std::unordered_map<Int, std::unordered_map<Int, Mat3x1>> adj_nodes_;
  std::unordered_map<Int, NodeInfo<Int>> nodes_m_to_c_;
  std::map<Int, std::map<Int, std::pair<Int, Int>>> local_cells_;
  std::vector<std::pair<Int, Int>> inner_adjs_;
  std::map<Int, std::vector<std::pair<Int, Int>>> part_to_adjs_;
  int rank_;

  Mat3x1 GetCoord(int zid, int nid) const {
    Mat3x1 coord;
    auto iter_zone = local_nodes_.find(zid);
    if (iter_zone != local_nodes_.end() && iter_zone->second.has(nid)) {
      coord[0] = iter_zone->second.x_[nid];
      coord[1] = iter_zone->second.y_[nid];
      coord[2] = iter_zone->second.z_[nid];
    } else {
      coord = adj_nodes_.at(zid).at(nid);
    }
    return coord;
  }
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_PARSER_HPP_
