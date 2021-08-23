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
#include "mini/integrator/hexa.hpp"

namespace mini {
namespace mesh {
namespace cgns {

template <class Object>
void print(Object&& obj) {
  std::cout << obj << '\n' << std::endl;
}

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
  CellInfo(CellInfo const&) = default;
  CellInfo& operator=(CellInfo const&) = default;
  CellInfo(CellInfo&&) noexcept = default;
  CellInfo& operator=(CellInfo&&) noexcept = default;
  ~CellInfo() noexcept = default;
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

template <typename Int, typename Real>
struct Cell;

template <typename Int = cgsize_t, typename Real = double>
struct Face {
  using CellPtr = Cell<Int, Real>*;
  CellPtr holder, sharer;
};

template <typename Int = cgsize_t, typename Real = double>
struct Cell {
  Int metis_id;
};

template <typename Int = cgsize_t, typename Real = double>
class Hexa : public Cell<Int, Real>, public integrator::Hexa<Real, 4, 4, 4> {
  using Integrator = integrator::Hexa<Real, 4, 4, 4>;

 public:
  using Integrator::Integrator;
};

template <typename Int = cgsize_t, typename Real = double>
class CellGroup {
  Int head_, size_;
  ShiftedVector<std::unique_ptr<Cell<Int, Real>>> data_;

 public:
  CellGroup(int head, int size)
      : head_(head), size_(size), data_(size, head) {
  }
  CellGroup() = default;
  CellGroup(CellGroup const&) = default;
  CellGroup(CellGroup&&) noexcept = default;
  CellGroup& operator=(CellGroup const&) = default;
  CellGroup& operator=(CellGroup &&) noexcept = default;
  ~CellGroup() noexcept = default;

  Int size() const {
    return size_;
  }
  bool has(int cid) const {
    return head_ <= cid && cid < size_ + head_;
  }
  auto& operator[](Int cell_id) {
    return data_[cell_id];
  }
};

template <typename Int = cgsize_t, typename Real = double>
class Parser{
  static constexpr int kLineWidth = 30;

 public:
  Parser(std::string const& cgns_file, std::string const& prefix, int pid) {
    rank_ = pid;
    int fid;
    if (cgp_open(cgns_file.c_str(), CG_MODE_READ, &fid))
      cgp_error_exit();
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
      auto& y = local_nodes_[zid].y_;
      auto& z = local_nodes_[zid].z_;
      if (cgp_coord_read_data(fid, 1, zid, 1, range_min, range_max, x.data()) ||
          cgp_coord_read_data(fid, 1, zid, 2, range_min, range_max, y.data()) ||
          cgp_coord_read_data(fid, 1, zid, 3, range_min, range_max, z.data()))
        cgp_error_exit();
      auto& metis_id = local_nodes_[zid].metis_id_;
      cgsize_t mem_dimensions[] = { tail - head };
      cgsize_t mem_range_min[] = { 1 };
      cgsize_t mem_range_max[] = { mem_dimensions[0] };
      if (cgp_field_general_read_data(fid, 1, zid, 1, 2, range_min, range_max,
          sizeof(Int) == 8 ? CGNS_ENUMV(LongInteger) : CGNS_ENUMV(Integer),
          1, mem_dimensions, mem_range_min, mem_range_max, metis_id.data()))
        cgp_error_exit();
      for (int nid = head; nid < tail; ++nid) {
        auto mid = metis_id[nid];
        nodes_m_to_c_[mid] = NodeInfo<Int>(zid, nid);
      }
    }
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
      int p, mid, zid, nid;
      std::sscanf(line, "%d %d %d %d", &p, &mid, &zid, &nid);
      recv_nodes[p].emplace_back(mid);
      nodes_m_to_c_[mid] = NodeInfo<Int>(zid, nid);
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
    requests.clear();
    // copy node coordinates from buffer to member
    int i_source = 0;
    for (auto& [source, nodes] : recv_nodes) {
      double* xyz = recv_bufs[i_source++].data();
      for (auto metis_id : nodes) {
        auto& info = nodes_m_to_c_[metis_id];
        adj_nodes_[info.zone_id][info.node_id] = { xyz[0], xyz[1] , xyz[2] };
        xyz += 3;
        // if (rank_ == 2) {
        //   int zid = info.zone_id, nid = info.node_id;
        //   std::cout << metis_id << ' ' << zid << ' ' << nid << ' ';
        //   print(adj_nodes_[zid][nid].transpose());
        // }
      }
    }
    // cell ranges
    std::map<Int, std::map<Int, ShiftedVector<Int>>> z_s_c_index;
    std::map<Int, std::map<Int, std::vector<Int>>> z_s_nodes;
    while (istrm.getline(line, 30) && line[0]) {
      int zid, sid, head, tail;
      std::sscanf(line, "%d %d %d %d", &zid, &sid, &head, &tail);
      // std::printf("zid = %4d, sid = %4d, head = %4d, tail = %4d\n",
      //       zid, sid, head, tail);
      local_cells_[zid][sid] = CellGroup<Int, Real>(head, tail - head);
      cgsize_t range_min[] = { head };
      cgsize_t range_max[] = { tail - 1 };
      cgsize_t mem_dimensions[] = { tail - head };
      cgsize_t mem_range_min[] = { 1 };
      cgsize_t mem_range_max[] = { mem_dimensions[0] };
      ShiftedVector<Int> metis_ids(mem_dimensions[0], head);
      char sol_name[33], field_name[33];
      GridLocation_t loc;
      DataType_t dt;
      if (cg_sol_info(fid, 1, zid, 2, sol_name, &loc) ||
          cg_field_info(fid, 1, zid, 2, 2, &dt, field_name))
        cgp_error_exit();
      std::cout << sol_name << ' ' << field_name << ' ' << mem_range_min[0] << ' ' << mem_range_max[0] << ' ' << mem_dimensions[0] << ' ' << metis_ids.size() << std::endl;
      if (cgp_field_general_read_data(fid, 1, zid, 2, 2, range_min, range_max,
          sizeof(Int) == 8 ? CGNS_ENUMV(LongInteger) : CGNS_ENUMV(Integer),
          1, mem_dimensions, mem_range_min, mem_range_max, metis_ids.data()))
        cgp_error_exit();
      for (int cid = head; cid < tail; ++cid) {
        auto mid = metis_ids[cid];
        cells_m_to_c_[mid] = CellInfo<Int>(zid, sid, cid);
        if (rank_ == 2)
          std::printf("mid = %ld, zid = %ld, sid = %ld, cid = %ld\n", mid,
              cells_m_to_c_[mid].zone_id, cells_m_to_c_[mid].section_id,
              cells_m_to_c_[mid].cell_id);
      }
      char name[33];
      ElementType_t type;
      cgsize_t u, v;
      int x, y;
      z_s_c_index[zid][sid] = ShiftedVector<Int>(mem_dimensions[0], head);
      auto& index = z_s_c_index[zid][sid];
      auto& nodes = z_s_nodes[zid][sid];
      if (cg_section_read(fid, 1, zid, sid, name, &type, &u, &v, &x, &y))
        cgp_error_exit();
      switch (type) {
      case CGNS_ENUMV(HEXA_8):
        nodes.resize(8 * mem_dimensions[0]);
        for (int i = 0; i < index.size(); ++i) {
          index[i] = 8 * i;
        }
        break;
      default:
        assert(false);
      }
      if (cgp_elements_read_data(fid, 1, zid, sid, range_min[0], range_max[0],
          nodes.data()))
        cgp_error_exit();
      for (int cid = head; cid < tail; ++cid) {
        int i = (cid - head) * 8;
        // if (rank_ == 2)
        //   std::printf("i = %4d, ", i);
        local_cells_[zid][sid][cid].reset(new Hexa<Int, Real>(
            GetCoord(zid, nodes[i+0]), GetCoord(zid, nodes[i+1]),
            GetCoord(zid, nodes[i+2]), GetCoord(zid, nodes[i+3]),
            GetCoord(zid, nodes[i+4]), GetCoord(zid, nodes[i+5]),
            GetCoord(zid, nodes[i+6]), GetCoord(zid, nodes[i+7])
        ));
        local_cells_[zid][sid][cid]->metis_id = metis_ids.at(cid);
        // if (rank_ == 2)
        //   std::printf("zid = %4d, sid = %4d, cid = %4d, mid = %4d\n",
        //       (int)zid, (int)sid, (int)cid, (int)metis_ids[cid]);
      }
    }
    // inner adjacency
    while (istrm.getline(line, 30) && line[0]) {
      int i, j;
      std::sscanf(line, "%d %d", &i, &j);
      inner_adjs_.emplace_back(i, j);
    }
    // interpart adjacency
    std::map<Int, std::map<Int, Int>> send_infos; // pid_to_mid_to_cnt
    std::map<Int, std::map<Int, Int>> recv_infos;
    std::vector<std::pair<Int, Int>> interpart_adjs;
    while (istrm.getline(line, 30) && line[0]) {
      int p, i, j, cnt_i, cnt_j;
      std::sscanf(line, "%d %d %d %d %d", &p, &i, &j, &cnt_i, &cnt_j);
      send_infos[p][i] = cnt_i;
      recv_infos[p][j] = cnt_j;
      interpart_adjs.emplace_back(i, j);
    }
    // send cell Info
    std::vector<std::vector<Real>> send_cells;
    for (auto& [target, cell_infos] : send_infos) {
      auto& send_buf = send_cells.emplace_back();
      for (auto& [mid, cnt] : cell_infos) {
        auto& info = cells_m_to_c_[mid];
        if (rank_ == 1)
          std::printf("info at %p\n", &info);
        Int zid = cells_m_to_c_[mid].zone_id,
            sid = cells_m_to_c_[mid].section_id,
            cid = cells_m_to_c_[mid].cell_id;
        auto id = z_s_c_index[zid][sid][cid];
        auto nodes = z_s_nodes[zid][sid];
        send_buf.emplace_back(zid);
        for (int i = 0; i < cnt; ++i) {
          send_buf.emplace_back(nodes[id+i]);
        }
      }
      int count = send_buf.size();
      MPI_Datatype datatype = sizeof(Int) == 8 ? MPI_LONG : MPI_INT;
      int tag = target;
      auto& req = requests.emplace_back();
      MPI_Isend(send_buf.data(), count, datatype, target, tag, MPI_COMM_WORLD,
          &req);
    }
    if (cgp_close(fid))
      cgp_error_exit();
  }

 private:
  using Mat3x1 = Eigen::Matrix<Real, 3, 1>;
  std::map<Int, NodeGroup<Int, Real>> local_nodes_;
  std::unordered_map<Int, std::unordered_map<Int, Mat3x1>> adj_nodes_;
  std::unordered_map<Int, NodeInfo<Int>> nodes_m_to_c_;
  std::unordered_map<Int, CellInfo<Int>> cells_m_to_c_;
  std::map<Int, std::map<Int, CellGroup<Int, Real>>> local_cells_;
  // metis_id to Cell
  std::unordered_map<Int, std::unique_ptr<Cell<Int, Real>>> ghost_cells_;
  std::vector<std::pair<Int, Int>> inner_adjs_;
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
