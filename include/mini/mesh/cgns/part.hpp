// Copyright 2021 PEI Weicheng and JIANG Yuyan
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
#include "mini/algebra/eigen.hpp"

#include "mini/mesh/cgns/format.hpp"
#include "mini/integrator/quad.hpp"
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
  NodeInfo(Int zi, Int ni) : i_zone(zi), i_node(ni) {}
  NodeInfo(NodeInfo const&) = default;
  NodeInfo& operator=(NodeInfo const&) = default;
  NodeInfo(NodeInfo&&) noexcept = default;
  NodeInfo& operator=(NodeInfo&&) noexcept = default;
  ~NodeInfo() noexcept = default;
  Int i_zone{0}, i_node{0};
};
template <class Int = int>
struct CellInfo {
  CellInfo() = default;
  CellInfo(Int zi, Int si, Int ci) : i_zone(zi), i_sect(si), i_cell(ci) {}
  CellInfo(CellInfo const&) = default;
  CellInfo& operator=(CellInfo const&) = default;
  CellInfo(CellInfo&&) noexcept = default;
  CellInfo& operator=(CellInfo&&) noexcept = default;
  ~CellInfo() noexcept = default;
  Int i_zone{0}, i_sect{0}, i_cell{0};
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
  bool has(int i_node) const {
    return head_ <= i_node && i_node < size_ + head_;
  }
};

template <typename Int, typename Real, int kFunc>
struct Cell;

template <typename Int = cgsize_t, typename Real = double, int kFunc = 2>
struct Face {
  using GaussPtr = std::unique_ptr<integrator::Face<Real, 3>>;
  using CellPtr = Cell<Int, Real, kFunc>*;
  GaussPtr gauss_;
  CellPtr holder_, sharer_;

  Face(GaussPtr&& gauss, CellPtr holder, CellPtr sharer)
      : gauss_(std::move(gauss)), holder_(holder), sharer_(sharer) {
  }
  Face(const Face&) = delete;
  Face& operator=(const Face&) = delete;
  Face(Face&&) noexcept = default;
  Face& operator=(Face&&) noexcept = default;
  ~Face() noexcept = default;
};

template <typename Int = cgsize_t, typename Real = double, int kFunc = 2>
struct Cell {
  static constexpr int kDim = 3;
  static constexpr int kOrder = 2;
  using Projection = integrator::Projection<Real, kDim, kOrder, kFunc>;
  using Basis = integrator::OrthoNormalBasis<Real, kDim, kOrder>;
  using GaussPtr = std::unique_ptr<integrator::Cell<Real>>;
  static constexpr int K = Projection::K;  // number of functions
  static constexpr int N = Projection::N;  // size of the basis

  Basis basis_;
  GaussPtr gauss_;
  Projection func_;
  Int metis_id;

  using Value = decltype(func_(gauss_->GetCenter()));

  Cell(GaussPtr&& gauss, Int m_cell)
      : basis_(*gauss), gauss_(std::move(gauss)), metis_id(m_cell) {
  }
  Cell() = default;
  Cell(const Cell&) = delete;
  Cell& operator=(const Cell&) = delete;
  Cell(Cell&&) noexcept = default;
  Cell& operator=(Cell&&) noexcept = default;
  ~Cell() noexcept = default;

  template <class Callable>
  void Project(Callable&& new_func) {
    func_.Project(new_func, basis_);
  }
};

template <typename Int = cgsize_t, typename Real = double, int kFunc = 2>
class CellGroup {
  using CellType = Cell<Int, Real, kFunc>;
  static constexpr int kFields = CellType::K * CellType::N;
  Int head_, size_;
  ShiftedVector<CellType> cells_;
  ShiftedVector<ShiftedVector<Real>> fields_;  // [i_field][i_cell]

 public:
  CellGroup(int head, int size)
      : head_(head), size_(size), cells_(size, head), fields_(kFields, 1) {
    for (int i = 1; i <= kFields; ++i) {
      fields_[i] = ShiftedVector<Real>(size, head);
    }
  }
  CellGroup() = default;
  CellGroup(CellGroup const&) = delete;
  CellGroup(CellGroup&&) noexcept = default;
  CellGroup& operator=(CellGroup const&) = delete;
  CellGroup& operator=(CellGroup &&) noexcept = default;
  ~CellGroup() noexcept = default;

  Int head() const {
    return head_;
  }
  Int size() const {
    return size_;
  }
  bool has(int i_cell) const {
    return head_ <= i_cell && i_cell < size_ + head_;
  }
  const CellType& operator[](Int i_cell) const {
    return cells_[i_cell];
  }
  CellType& operator[](Int i_cell) {
    return cells_[i_cell];
  }
  const ShiftedVector<Real>& GetField(Int i_field) const {
    return fields_[i_field];
  }
  void GatherFields() {
    for (int i_cell = head(); i_cell < head() + size(); ++i_cell) {
      const auto& cell = cells_.at(i_cell);
      const auto& coeff = cell.func_.GetCoeff();
      for (int i_field = 1; i_field <= kFields; ++i_field) {
        fields_.at(i_field).at(i_cell) = coeff.reshaped()[i_field-1];
      }
    }
  }
};

template <typename Int = cgsize_t, typename Real = double, int kFunc = 2>
class Part {
  static constexpr int kLineWidth = 30;
  static constexpr int kDim = 3;
  static constexpr int kFields = kFunc * Cell<Int, Real, kFunc>::N;
  static constexpr int i_base = 1;
  static constexpr auto kIntType
      = sizeof(Int) == 8 ? CGNS_ENUMV(LongInteger) : CGNS_ENUMV(Integer);
  static constexpr auto kRealType
      = sizeof(Real) == 8 ? CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
  static const MPI_Datatype kMpiIntType;
  static const MPI_Datatype kMpiRealType;

 public:
  Part(std::string const& directory, int rank)
      : directory_(directory), cgns_file_(directory + "/whole/shuffled.cgns"),
        part_path_(directory + "/parts/"), rank_(rank) {
    int i_file;
    if (cgp_open(cgns_file_.c_str(), CG_MODE_READ, &i_file))
      cgp_error_exit();
    auto txt_file = part_path_ + std::to_string(rank) + ".txt";
    auto istrm = std::ifstream(txt_file);
    char line[kLineWidth];
    // node ranges
    while (istrm.getline(line, 30) && line[0]) {
      int i_zone, head, tail;
      std::sscanf(line, "%d %d %d", &i_zone, &head, &tail);
      local_nodes_[i_zone] = NodeGroup<Int, Real>(head, tail - head);
      cgsize_t range_min[] = { head };
      cgsize_t range_max[] = { tail - 1 };
      auto& x = local_nodes_[i_zone].x_;
      auto& y = local_nodes_[i_zone].y_;
      auto& z = local_nodes_[i_zone].z_;
      if (cgp_coord_read_data(i_file, i_base, i_zone, 1,
          range_min, range_max, x.data()) ||
          cgp_coord_read_data(i_file, i_base, i_zone, 2,
          range_min, range_max, y.data()) ||
          cgp_coord_read_data(i_file, i_base, i_zone, 3,
          range_min, range_max, z.data()))
        cgp_error_exit();
      auto& metis_id = local_nodes_[i_zone].metis_id_;
      cgsize_t mem_dimensions[] = { tail - head };
      cgsize_t mem_range_min[] = { 1 };
      cgsize_t mem_range_max[] = { mem_dimensions[0] };
      if (cgp_field_general_read_data(i_file, i_base, i_zone, 1, 2,
          range_min, range_max, kIntType,
          1, mem_dimensions, mem_range_min, mem_range_max, metis_id.data()))
        cgp_error_exit();
      for (int i_node = head; i_node < tail; ++i_node) {
        auto m_node = metis_id[i_node];
        m_to_node_info_[m_node] = NodeInfo<Int>(i_zone, i_node);
      }
    }
    // send nodes info
    std::map<Int, std::vector<Int>> send_nodes;
    while (istrm.getline(line, 30) && line[0]) {
      int i_part, m_node;
      std::sscanf(line, "%d %d", &i_part, &m_node);
      send_nodes[i_part].emplace_back(m_node);
    }
    std::vector<MPI_Request> requests;
    std::vector<std::vector<Real>> send_bufs;
    for (auto& [i_part, nodes] : send_nodes) {
      auto& buf = send_bufs.emplace_back();
      for (auto m_node : nodes) {
        auto& info = m_to_node_info_[m_node];
        auto const& coord = GetCoord(info.i_zone, info.i_node);
        buf.emplace_back(coord[0]);
        buf.emplace_back(coord[1]);
        buf.emplace_back(coord[2]);
      }
      assert(std::is_sorted(nodes.begin(), nodes.end()));
      int count = 3 * nodes.size();
      int tag = i_part;
      auto& request = requests.emplace_back();
      MPI_Isend(buf.data(), count, kMpiRealType, i_part, tag, MPI_COMM_WORLD,
          &request);
    }
    // recv nodes info
    std::map<Int, std::vector<Int>> recv_nodes;
    while (istrm.getline(line, 30) && line[0]) {
      int i_part, m_node, i_zone, i_node;
      std::sscanf(line, "%d %d %d %d", &i_part, &m_node, &i_zone, &i_node);
      recv_nodes[i_part].emplace_back(m_node);
      m_to_node_info_[m_node] = NodeInfo<Int>(i_zone, i_node);
    }
    std::vector<std::vector<Real>> recv_bufs;
    for (auto& [i_part, nodes] : recv_nodes) {
      assert(std::is_sorted(nodes.begin(), nodes.end()));
      int count = 3 * nodes.size();
      auto& buf = recv_bufs.emplace_back(std::vector<Real>(count));
      int tag = rank_;
      auto& request = requests.emplace_back();
      MPI_Irecv(buf.data(), count, kMpiRealType, i_part, tag, MPI_COMM_WORLD,
          &request);
    }
    // wait until all send/recv finish
    std::vector<MPI_Status> statuses(requests.size());
    MPI_Waitall(requests.size(), requests.data(), statuses.data());
    requests.clear();
    // copy node coordinates from buffer to member
    int i_source = 0;
    for (auto& [i_part, nodes] : recv_nodes) {
      double* xyz = recv_bufs[i_source++].data();
      for (auto m_node : nodes) {
        auto& info = m_to_node_info_[m_node];
        ghost_nodes_[info.i_zone][info.i_node] = { xyz[0], xyz[1] , xyz[2] };
        xyz += 3;
        if (rank_ == -1) {
          int i_zone = info.i_zone, i_node = info.i_node;
          std::cout << m_node << ' ' << i_zone << ' ' << i_node << ' ';
          print(ghost_nodes_[i_zone][i_node].transpose());
        }
      }
    }
    // cell ranges
    std::map<Int, std::map<Int, ShiftedVector<Int>>> z_s_c_index;
    std::map<Int, std::map<Int, std::vector<Int>>> z_s_nodes;
    while (istrm.getline(line, 30) && line[0]) {
      int i_zone, i_sect, head, tail;
      std::sscanf(line, "%d %d %d %d", &i_zone, &i_sect, &head, &tail);
      if (rank_ == -1)
        std::printf("i_zone = %4d, i_sect = %4d, head = %4d, tail = %4d\n",
            i_zone, i_sect, head, tail);
      local_cells_[i_zone][i_sect] = CellGroup<Int, Real>(head, tail - head);
      cgsize_t range_min[] = { head };
      cgsize_t range_max[] = { tail - 1 };
      cgsize_t mem_dimensions[] = { tail - head };
      cgsize_t mem_range_min[] = { 1 };
      cgsize_t mem_range_max[] = { mem_dimensions[0] };
      ShiftedVector<Int> metis_ids(mem_dimensions[0], head);
      char sol_name[33], field_name[33];
      GridLocation_t loc;
      DataType_t dt;
      if (cg_sol_info(i_file, i_base, i_zone, 2, sol_name, &loc) ||
          cg_field_info(i_file, i_base, i_zone, 2, 2, &dt, field_name))
        cgp_error_exit();
      if (rank_ == -1)
        std::cout << sol_name << ' ' << field_name << std::endl;
      if (cgp_field_general_read_data(i_file, i_base, i_zone, 2, 2,
          range_min, range_max, kIntType,
          1, mem_dimensions, mem_range_min, mem_range_max, metis_ids.data()))
        cgp_error_exit();
      for (int i_cell = head; i_cell < tail; ++i_cell) {
        auto m_cell = metis_ids[i_cell];
        m_to_cell_info_[m_cell] = CellInfo<Int>(i_zone, i_sect, i_cell);
        if (rank_ == -1) {
          std::printf("m_cell = %ld, ", m_cell);
          std::printf("i_zone = %ld, i_sect = %ld, i_cell = %ld\n",
              m_to_cell_info_[m_cell].i_zone,
              m_to_cell_info_[m_cell].i_sect,
              m_to_cell_info_[m_cell].i_cell);
        }
      }
      char name[33];
      ElementType_t type;
      cgsize_t u, v;
      int x, y;
      z_s_c_index[i_zone][i_sect] = ShiftedVector<Int>(mem_dimensions[0], head);
      auto& index = z_s_c_index[i_zone][i_sect];
      auto& nodes = z_s_nodes[i_zone][i_sect];
      if (cg_section_read(i_file, i_base, i_zone, i_sect, name, &type,
          &u, &v, &x, &y))
        cgp_error_exit();
      switch (type) {
      case CGNS_ENUMV(HEXA_8):
        nodes.resize(8 * mem_dimensions[0]);
        for (int i = 0; i < index.size(); ++i) {
          index.at(head + i) = 8 * i;
        }
        break;
      default:
        assert(false);
      }
      if (cgp_elements_read_data(i_file, i_base, i_zone, i_sect,
          range_min[0], range_max[0], nodes.data()))
        cgp_error_exit();
      for (int i_cell = head; i_cell < tail; ++i_cell) {
        int i = (i_cell - head) * 8;
        auto hexa_ptr = std::make_unique<integrator::Hexa<Real, 4, 4, 4>>(
            GetCoord(i_zone, nodes[i+0]), GetCoord(i_zone, nodes[i+1]),
            GetCoord(i_zone, nodes[i+2]), GetCoord(i_zone, nodes[i+3]),
            GetCoord(i_zone, nodes[i+4]), GetCoord(i_zone, nodes[i+5]),
            GetCoord(i_zone, nodes[i+6]), GetCoord(i_zone, nodes[i+7]));
        auto cell = Cell<Int, Real, kFunc>(
            std::move(hexa_ptr), metis_ids[i_cell]);
        local_cells_[i_zone][i_sect][i_cell] = std::move(cell);
        if (rank_ == -1) {
          std::cout << "i_zone = " << i_zone << ", i_sect = " << i_sect
              << ", i_cell = " << i_cell << ", m_cell = " << metis_ids[i_cell];
          std::cout << std::endl;
        }
      }
    }
    // local adjacency
    while (istrm.getline(line, 30) && line[0]) {
      int i, j;
      std::sscanf(line, "%d %d", &i, &j);
      local_adjs_.emplace_back(i, j);
    }
    // ghost adjacency
    std::map<Int, std::map<Int, Int>>
        send_infos, recv_infos;  // [i_part][m_cell] -> cnt
    std::vector<std::pair<Int, Int>> ghost_adjs;
    while (istrm.getline(line, 30) && line[0]) {
      int p, i, j, cnt_i, cnt_j;
      std::sscanf(line, "%d %d %d %d %d", &p, &i, &j, &cnt_i, &cnt_j);
      send_infos[p][i] = cnt_i;
      recv_infos[p][j] = cnt_j;
      ghost_adjs.emplace_back(i, j);
    }
    // send cell info
    std::vector<std::vector<Int>> send_cells;
    for (auto& [i_part, cell_infos] : send_infos) {
      auto& send_buf = send_cells.emplace_back();
      for (auto& [m_cell, cnt] : cell_infos) {
        auto& info = m_to_cell_info_[m_cell];
        if (rank_ == -1)
          std::printf("info at %p\n", &info);
        Int i_zone = m_to_cell_info_[m_cell].i_zone,
            i_sect = m_to_cell_info_[m_cell].i_sect,
            i_cell = m_to_cell_info_[m_cell].i_cell;
        auto id = z_s_c_index[i_zone][i_sect][i_cell];
        auto nodes = z_s_nodes[i_zone][i_sect];
        send_buf.emplace_back(i_zone);
        for (int i = 0; i < cnt; ++i) {
          send_buf.emplace_back(nodes[id+i]);
        }
      }
      int count = send_buf.size();
      int tag = i_part;
      auto& req = requests.emplace_back();
      MPI_Isend(send_buf.data(), count, kMpiIntType, i_part, tag,
          MPI_COMM_WORLD, &req);
    }
    // recv cell info
    std::unordered_map<Int, std::pair<int, int>> m_to_recv_cells;
    std::vector<std::vector<Int>> recv_cells;
    for (auto& [i_part, cell_infos] : recv_infos) {
      auto& buf = recv_cells.emplace_back();
      int count = 0;
      for (auto& [m_cell, cnt] : cell_infos) {
        ++count;
        count += cnt;
      }
      int tag = rank_;
      buf.resize(count);
      auto& req = requests.emplace_back();
      MPI_Irecv(buf.data(), count, kMpiIntType, i_part, tag, MPI_COMM_WORLD,
          &req);
    }
    // wait until all send/recv finish
    statuses = std::vector<MPI_Status>(requests.size());
    MPI_Waitall(requests.size(), requests.data(), statuses.data());
    requests.clear();
    // build ghost cells
    int i_buf = 0;
    for (auto& [i_part, cell_infos] : recv_infos) {
      auto& buf = recv_cells.at(i_buf);
      int id = 0;
      for (auto& [m_cell, cnt] : cell_infos) {
        m_to_recv_cells[m_cell].first = i_buf;
        m_to_recv_cells[m_cell].second = id+1;
        int i_zone = buf[id++];
        auto hexa_ptr = std::make_unique<integrator::Hexa<Real, 4, 4, 4>>(
          GetCoord(i_zone, buf[id+0]), GetCoord(i_zone, buf[id+1]),
          GetCoord(i_zone, buf[id+2]), GetCoord(i_zone, buf[id+3]),
          GetCoord(i_zone, buf[id+4]), GetCoord(i_zone, buf[id+5]),
          GetCoord(i_zone, buf[id+6]), GetCoord(i_zone, buf[id+7]));
        ghost_cells_[m_cell] = Cell<Int, Real, kFunc>(
            std::move(hexa_ptr), m_cell);
        id += cnt;
      }
      ++i_buf;
    }
    // build local faces
    for (auto [m_holder, m_sharer] : local_adjs_) {
      auto& holder_info = m_to_cell_info_[m_holder];
      auto& sharer_info = m_to_cell_info_[m_sharer];
      auto i_zone = holder_info.i_zone;
      // find the common nodes of the holder and the sharer
      auto m_count = std::unordered_map<Int, Int>();
      auto& holder_nodes = z_s_nodes[i_zone][holder_info.i_sect];
      auto& sharer_nodes = z_s_nodes[i_zone][sharer_info.i_sect];
      auto holder_head =
          z_s_c_index[i_zone][holder_info.i_sect][holder_info.i_cell];
      auto sharer_head =
          z_s_c_index[i_zone][sharer_info.i_sect][sharer_info.i_cell];
      for (int i = 0; i < 8; ++i) {
        ++m_count[holder_nodes[holder_head + i]];
        ++m_count[sharer_nodes[sharer_head + i]];
      }
      auto common_nodes = std::vector<Int>();
      common_nodes.reserve(4);
      for (auto [i_cell, cnt] : m_count)
        if (cnt == 2)
          common_nodes.emplace_back(i_cell);
      assert(common_nodes.size() == 4);
      // let the normal vector point from holder to sharer
      // see http://cgns.github.io/CGNS_docs_current/sids/conv.figs/hexa_8.png
      auto& zone = local_cells_[i_zone];
      auto& holder = zone[holder_info.i_sect][holder_info.i_cell];
      auto& sharer = zone[sharer_info.i_sect][sharer_info.i_cell];
      integrator::Hexa<Real, 4, 4, 4>::SortNodesOnFace(
          4, &holder_nodes[holder_head], common_nodes.data());
      if (rank_ == -1)
        std::cout << "Quad{ " << common_nodes[0] << ", " << common_nodes[1]
            << ", " << common_nodes[2] << ", " << common_nodes[3] << " }\n";
      // build the quad integrator
      auto quad_ptr = std::make_unique<integrator::Quad<Real, kDim, 4, 4>>(
          GetCoord(i_zone, common_nodes[0]), GetCoord(i_zone, common_nodes[1]),
          GetCoord(i_zone, common_nodes[2]), GetCoord(i_zone, common_nodes[3]));
      local_faces_.emplace_back(std::move(quad_ptr), &holder, &sharer);
    }
    std::cout << local_faces_.size() << " local faces in rank " << rank_;
    std::cout << std::endl;
    // build ghost faces
    for (auto [m_holder, m_sharer] : ghost_adjs) {
      auto& holder_info = m_to_cell_info_[m_holder];
      auto& sharer_info = m_to_recv_cells[m_sharer];
      auto i_zone = holder_info.i_zone;
      // find the common nodes of the holder and the sharer
      auto m_count = std::unordered_map<Int, Int>();
      auto& holder_nodes = z_s_nodes[i_zone][holder_info.i_sect];
      auto& sharer_nodes = recv_cells[sharer_info.first];
      auto holder_head =
          z_s_c_index[i_zone][holder_info.i_sect][holder_info.i_cell];
      auto sharer_head = sharer_info.second;
      for (int i = 0; i < 8; ++i) {
        ++m_count[holder_nodes[holder_head + i]];
        ++m_count[sharer_nodes[sharer_head + i]];
      }
      auto common_nodes = std::vector<Int>();
      common_nodes.reserve(4);
      for (auto [i_cell, cnt] : m_count) {
        if (cnt == 2)
          common_nodes.emplace_back(i_cell);
      }
      assert(common_nodes.size() == 4);
      // let the normal vector point from holder to sharer
      // see http://cgns.github.io/CGNS_docs_current/sids/conv.figs/hexa_8.png
      auto& zone = local_cells_[i_zone];
      auto& holder = zone[holder_info.i_sect][holder_info.i_cell];
      auto& sharer = ghost_cells_[m_sharer];
      integrator::Hexa<Real, 4, 4, 4>::SortNodesOnFace(
          4, &holder_nodes[holder_head], common_nodes.data());
      if (rank_ == -1)
        std::cout << "Quad{ " << common_nodes[0] << ", " << common_nodes[1]
            << ", " << common_nodes[2] << ", " << common_nodes[3] << " }\n";
      // build the quad integrator
      auto quad_ptr = std::make_unique<integrator::Quad<Real, kDim, 4, 4>>(
          GetCoord(i_zone, common_nodes[0]), GetCoord(i_zone, common_nodes[1]),
          GetCoord(i_zone, common_nodes[2]), GetCoord(i_zone, common_nodes[3]));
      if (m_holder < m_sharer)
        ghost_faces_.emplace_back(std::move(quad_ptr), &holder, &sharer);
      else
        ghost_faces_.emplace_back(std::move(quad_ptr), &sharer, &holder);
    }
    std::cout << local_faces_.size() << " ghost faces in rank "
        << rank_ << std::endl;
    if (cgp_close(i_file))
      cgp_error_exit();
  }
  template <class Callable>
  void Project(Callable&& new_func) {
    for (auto& [i_zone, sects] : local_cells_) {
      for (auto& [i_sect, cells] : sects) {
        auto tail = cells.head() + cells.size();
        for (Int i_cell = cells.head(); i_cell < tail; ++i_cell) {
          cells[i_cell].Project(new_func);
        }
      }
    }
  }
  void GatherSolutions() {
    int n_zones = local_nodes_.size();
    for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
      int n_sects = local_cells_[i_zone].size();
      for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
        local_cells_[i_zone][i_sect].GatherFields();
      }
    }
  }
  void WriteSolutions() const {
    int n_zones = local_nodes_.size();
    int i_file;
    if (cgp_open(cgns_file_.c_str(), CG_MODE_MODIFY, &i_file))
      cgp_error_exit();
    for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
      auto& zone = local_cells_.at(i_zone);
      int n_solns;
      if (cg_nsols(i_file, i_base, i_zone, &n_solns))
        cgp_error_exit();
      int i_soln;
      if (cg_sol_write(i_file, i_base, i_zone, "Solution0", CellCenter,
          &i_soln))
        cgp_error_exit();
      int n_fields = kFields;
      for (int i_field = 1; i_field <= n_fields; ++i_field) {
        int n_sects = zone.size();
        for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
          auto& section = zone.at(i_sect);
          auto name = "Field" + std::to_string(i_field);
          int field_id;
          if (cgp_field_write(i_file, i_base, i_zone, i_soln, kRealType,
              name.c_str(),  &field_id))
            cgp_error_exit();
          assert(field_id == i_field);
          cgsize_t first[] = { section.head() };
          cgsize_t last[] = { section.head() + section.size() - 1 };
          if (cgp_field_write_data(i_file, i_base, i_zone, i_soln, i_field,
              first, last, section.GetField(i_field).data()))
            cgp_error_exit();
        }
      }
    }
    if (cgp_close(i_file))
      cgp_error_exit();
  }
  void WriteSolutionsOnGaussPoints() const {
    auto vtk_file = part_path_ + std::to_string(rank_) + ".vtk";
    auto ostrm = std::ofstream(vtk_file);
    ostrm << "# vtk DataFile Version 2.0\n";
    ostrm << "Field values on quadrature points.\n";
    ostrm << "ASCII\n";
    ostrm << "DATASET UNSTRUCTURED_GRID\n";
    int n_points = 0;
    auto coords = std::vector<Mat3x1>();
    auto fields = std::vector<typename Cell<Int, Real, kFunc>::Value>();
    for (auto& [i_zone, zone] : local_cells_) {
      for (auto& [i_sect, sect] : zone) {
        int i_cell = sect.head();
        int i_cell_tail = sect.head() + sect.size();
        while (i_cell < i_cell_tail) {
          auto& cell_ptr = sect[i_cell].gauss_;
          for (int q = 0; q < cell_ptr->CountQuadPoints(); ++q) {
            coords.emplace_back(cell_ptr->GetGlobalCoord(q));
            fields.emplace_back(sect[i_cell].func_(coords.back()));
          }
          ++i_cell;
        }
      }
    }
    ostrm << "POINTS " << coords.size() << " double\n";
    for (auto& xyz : coords) {
      ostrm << xyz[0] << ' ' << xyz[1] << ' ' << xyz[2] << '\n';
    }
    ostrm << "CELLS " << coords.size() << ' ' << coords.size() * 2 << '\n';
    for (int i = 0; i < coords.size(); ++i) {
      ostrm << "1 " << i << '\n';
    }
    ostrm << "CELL_TYPES " << coords.size() << '\n';
    for (int i = 0; i < coords.size(); ++i) {
      ostrm << "1\n";
    }
    ostrm << "POINT_DATA " << coords.size() << "\n";
    int K = fields[0].size();
    for (int k = 0; k < K; ++k) {
      ostrm << "SCALARS Field[" << k + 1 << "] double 1\n";
      ostrm << "LOOKUP_TABLE field\n";
      for (auto& f : fields) {
        ostrm << f[k] << '\n';
      }
    }
  }
  void WriteSolutionsOnCellCenters() const {
    auto vtk_file = part_path_ + std::to_string(rank_) + ".vtk";
    auto ostrm = std::ofstream(vtk_file);
    ostrm << "# vtk DataFile Version 2.0\n";
    ostrm << "Field values on each cell.\n";
    ostrm << "ASCII\n";
    ostrm << "DATASET UNSTRUCTURED_GRID\n";
    int n_points = 0;
    auto coords = std::vector<Mat3x1>();
    auto cells = std::vector<Int>();
    auto fields = std::vector<typename Cell<Int, Real, kFunc>::Value>();
    for (auto& [i_zone, zone] : local_cells_) {
      for (auto& [i_sect, sect] : zone) {
        int i_cell = sect.head();
        int i_cell_tail = sect.head() + sect.size();
        while (i_cell < i_cell_tail) {
          cells.emplace_back(20);
          auto& gauss = sect[i_cell].gauss_;
          auto& func = sect[i_cell].func_;
          // nodes at corners
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({-1, -1, -1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({+1, -1, -1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({+1, +1, -1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({-1, +1, -1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({-1, -1, +1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({+1, -1, +1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({+1, +1, +1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({-1, +1, +1}));
          fields.emplace_back(func(coords.back()));
          // nodes on edges
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({0., -1, -1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({+1, 0., -1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({0., +1, -1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({-1, 0., -1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({0., -1, +1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({+1, 0., +1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({0., +1, +1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({-1, 0., +1}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({-1, -1, 0.}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({+1, -1, 0.}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({+1, +1, 0.}));
          fields.emplace_back(func(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss->LocalToGlobal({-1, +1, 0.}));
          fields.emplace_back(func(coords.back()));
          ++i_cell;
        }
      }
    }
    ostrm << "POINTS " << coords.size() << " double\n";
    for (auto& xyz : coords) {
      ostrm << xyz[0] << ' ' << xyz[1] << ' ' << xyz[2] << '\n';
    }
    auto n_cells = cells.size() / 21;
    ostrm << "CELLS " << n_cells << ' ' << cells.size() << '\n';
    for (auto c : cells) {
      ostrm << c << ' ';
    }
    ostrm << '\n';
    ostrm << "CELL_TYPES " << n_cells << '\n';
    for (int i = 0; i < n_cells; ++i) {
      ostrm << "25 ";
    }
    ostrm << '\n';
    ostrm << "POINT_DATA " << coords.size() << "\n";
    int K = fields[0].size();
    for (int k = 0; k < K; ++k) {
      ostrm << "SCALARS Field[" << k + 1 << "] double 1\n";
      ostrm << "LOOKUP_TABLE field\n";
      for (auto& f : fields) {
        ostrm << f[k] << '\n';
      }
      ostrm << '\n';
    }
  }

 private:
  using Mat3x1 = algebra::Matrix<Real, 3, 1>;
  std::map<Int, NodeGroup<Int, Real>>
      local_nodes_;  // [i_zone] -> a NodeGroup obj
  std::unordered_map<Int, std::unordered_map<Int, Mat3x1>>
      ghost_nodes_;  // [i_zone][i_node] -> a Mat3x1 obj
  std::unordered_map<Int, NodeInfo<Int>>
      m_to_node_info_;  // [m_node] -> a NodeInfo obj
  std::unordered_map<Int, CellInfo<Int>>
      m_to_cell_info_;  // [m_cell] -> a CellInfo obj
  std::map<Int, std::map<Int, CellGroup<Int, Real, kFunc>>>
      local_cells_;  // [i_zone][i_sect][i_cell] -> a Cell obj
  std::unordered_map<Int, Cell<Int, Real, kFunc>>
      ghost_cells_;  //                 [m_cell] -> a Cell obj
  std::vector<std::pair<Int, Int>>
      local_adjs_;  // [i_pair] -> { m_holder, m_sharer }
  std::vector<Face<Int, Real, kFunc>>
      local_faces_, ghost_faces_;  // [i_face] -> a Face obj
  const std::string directory_;
  const std::string cgns_file_;
  const std::string part_path_;
  int rank_;

  Mat3x1 GetCoord(int i_zone, int i_node) const {
    Mat3x1 coord;
    auto iter_zone = local_nodes_.find(i_zone);
    if (iter_zone != local_nodes_.end() && iter_zone->second.has(i_node)) {
      coord[0] = iter_zone->second.x_[i_node];
      coord[1] = iter_zone->second.y_[i_node];
      coord[2] = iter_zone->second.z_[i_node];
    } else {
      if (rank_ == -1) {
        std::cout << "i_zone = " << i_zone << ", i_node = " << i_node;
        std::cout << std::endl;
      }
      coord = ghost_nodes_.at(i_zone).at(i_node);
    }
    return coord;
  }
};
template <typename Int, typename Real, int kFunc>
MPI_Datatype const Part<Int, Real, kFunc>::kMpiIntType
    = sizeof(Int) == 8 ? MPI_LONG : MPI_INT;
template <typename Int, typename Real, int kFunc>
MPI_Datatype const Part<Int, Real, kFunc>::kMpiRealType
    = sizeof(Real) == 8 ? MPI_DOUBLE : MPI_FLOAT;

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_PARSER_HPP_
