// Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_MESH_PART_HPP_
#define MINI_MESH_PART_HPP_

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

#include "mpi.h"
#include "pcgnslib.h"
#include "mini/algebra/eigen.hpp"
#include "mini/mesh/cgns.hpp"
#include "mini/integrator/quad.hpp"
#include "mini/integrator/hexa.hpp"
#include "mini/polynomial/basis.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"

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
template <class Int = cgsize_t>
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

template <typename Int, typename Real, int kFunc, int kDim, int kOrder>
struct Cell;

template <typename Int = cgsize_t, typename Real = double, int kFunc = 2, int kDim = 3, int kOrder = 2>
struct Face {
  using GaussPtr = std::unique_ptr<integrator::Face<Real, 3>>;
  using CellPtr = Cell<Int, Real, kFunc, kDim, kOrder> *;
  // TODO(PVC): move Riemann out
  using Gas = mini::riemann::euler::IdealGas<1, 4>;
  using Solver = mini::riemann::euler::Exact<Gas, 3>;
  using Riemann = mini::riemann::rotated::Euler<Solver, 3>;

  GaussPtr gauss_ptr_;
  CellPtr holder_, sharer_;
  std::vector<Riemann> riemanns_;
  Int id_{-1};

  Face(GaussPtr&& gauss_ptr, CellPtr holder, CellPtr sharer, Int id = 0)
      : gauss_ptr_(std::move(gauss_ptr)), holder_(holder), sharer_(sharer),
        id_(id) {
    riemanns_.resize(gauss_ptr_->CountQuadPoints());
  }
  Face(const Face&) = delete;
  Face& operator=(const Face&) = delete;
  Face(Face&&) noexcept = default;
  Face& operator=(Face&&) noexcept = default;
  ~Face() noexcept = default;

  void RotateRiemanns() {
    for (int q = 0; q < gauss_ptr_->CountQuadPoints(); ++q) {
      auto& frame = gauss_ptr_->GetNormalFrame(q);
      riemanns_[q].Rotate(frame);
    }
  }
  const Riemann& GetRiemann(int i) const {
    return riemanns_[i];
  }
  Riemann& GetRiemann(int i) {
    return riemanns_[i];
  }
  Int id() const {
    return id_;
  }
};

template <typename Int = cgsize_t, typename Real = double, int kFunc = 2, int kDim = 3, int kOrder = 2>
struct Cell {
  using GaussPtr = std::unique_ptr<integrator::Cell<Real>>;
  using Basis = polynomial::OrthoNormal<Real, kDim, kOrder>;
  using Projection = polynomial::Projection<Real, kDim, kOrder, kFunc>;
  using Coord = typename Projection::Coord;
  using Value = typename Projection::Value;
  using Coeff = typename Projection::Coeff;
  static constexpr int K = Projection::K;  // number of functions
  static constexpr int N = Projection::N;  // size of the basis
  using MyFace = Face<Int, Real, kFunc, kDim, kOrder>;

  std::vector<Cell*> adj_cells_;
  std::vector<MyFace*> adj_faces_;
  Basis basis_;
  GaussPtr gauss_ptr_;
  Projection projection_;
  Int metis_id{-1}, id_{-1};
  bool inner_ = true;

  Cell(GaussPtr&& gauss_ptr, Int m_cell)
      : basis_(*gauss_ptr), gauss_ptr_(std::move(gauss_ptr)),
        metis_id(m_cell), projection_(basis_) {
  }
  Cell() = default;
  Cell(const Cell&) = delete;
  Cell& operator=(const Cell&) = delete;
  Cell(Cell&&) noexcept = default;
  Cell& operator=(Cell&&) noexcept = default;
  ~Cell() noexcept = default;

  Real volume() const {
    return gauss_ptr_->volume();
  }
  Int id() const {
    return id_;
  }
  bool inner() const {
    return inner_;
  }
  const Coord& center() const {
    return basis_.center();
  }

  template <class Callable>
  void Project(Callable&& func) {
    projection_.Project(func, basis_);
  }
};

template <typename Int = cgsize_t, typename Real = double, int kFunc = 2, int kDim = 3, int kOrder = 2>
class CellGroup {
  using CellType = Cell<Int, Real, kFunc, kDim, kOrder>;
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
  Int tail() const {
    return head_ + size_;
  }
  bool has(int i_cell) const {
    return head() <= i_cell && i_cell < tail();
  }
  const CellType& operator[](Int i_cell) const {
    return cells_[i_cell];
  }
  CellType& operator[](Int i_cell) {
    return cells_[i_cell];
  }
  const CellType& at(Int i_cell) const {
    return cells_.at(i_cell);
  }
  CellType& at(Int i_cell) {
    return cells_.at(i_cell);
  }
  auto begin() {
    return cells_.begin();
  }
  auto end() {
    return cells_.end();
  }
  auto begin() const {
    return cells_.begin();
  }
  auto end() const {
    return cells_.end();
  }
  auto cbegin() const {
    return cells_.cbegin();
  }
  auto cend() const {
    return cells_.cend();
  }
  const ShiftedVector<Real>& GetField(Int i_field) const {
    return fields_[i_field];
  }
  void GatherFields() {
    for (int i_cell = head(); i_cell < tail(); ++i_cell) {
      const auto& cell = cells_.at(i_cell);
      const auto& coeff = cell.projection_.coeff();
      for (int i_field = 1; i_field <= kFields; ++i_field) {
        fields_.at(i_field).at(i_cell) = coeff.reshaped()[i_field-1];
      }
    }
  }
};

template <typename Int = cgsize_t, typename Real = double, int kFunc = 2, int kDim = 3, int kOrder = 2>
class Part {
 public:
  using FaceType = Face<Int, Real, kFunc, kDim, kOrder>;
  using CellType = Cell<Int, Real, kFunc, kDim, kOrder>;

 private:
  using CellGroupType = CellGroup<Int, Real, kFunc, kDim, kOrder>;
  using CellPtr = CellType *;
  static constexpr int kLineWidth = 128;
  static constexpr int kFields = kFunc * CellType::N;
  static constexpr int i_base = 1;
  static constexpr auto kIntType
      = sizeof(Int) == 8 ? CGNS_ENUMV(LongInteger) : CGNS_ENUMV(Integer);
  static constexpr auto kRealType
      = sizeof(Real) == 8 ? CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
  static const MPI_Datatype kMpiIntType;
  static const MPI_Datatype kMpiRealType;

 public:
  Part(std::string const& directory, int rank)
      : directory_(directory), cgns_file_(directory + "/shuffled.cgns"),
        part_path_(directory + "/partition/"), rank_(rank) {
    int i_file;
    if (cgp_open(cgns_file_.c_str(), CG_MODE_READ, &i_file))
      cgp_error_exit();
    auto txt_file = part_path_ + std::to_string(rank) + ".txt";
    auto istrm = std::ifstream(txt_file);
    BuildLocalNodes(istrm, i_file);
    auto [recv_nodes, recv_coords] = ShareGhostNodes(istrm);
    BuildGhostNodes(recv_nodes, recv_coords);
    auto cell_conn = BuildLocalCells(istrm, i_file);
    auto ghost_adj = BuildAdj(istrm);
    auto recv_cells = ShareGhostCells(ghost_adj, cell_conn);
    auto m_to_recv_cells = BuildGhostCells(ghost_adj, recv_cells);
    FillCellPtrs(ghost_adj);
    AddLocalCellId();
    BuildLocalFaces(cell_conn);
    BuildGhostFaces(ghost_adj, cell_conn, recv_cells, m_to_recv_cells);
    BuildBoundaryFaces(cell_conn, istrm, i_file);
    if (cgp_close(i_file))
      cgp_error_exit();
  }

 private:
  void BuildLocalNodes(std::ifstream& istrm, int i_file) {
    char line[kLineWidth];
    istrm.getline(line, kLineWidth); assert(line[0] == '#');
    // node coordinates
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
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
  }
  std::pair<
    std::map<Int, std::vector<Int>>,
    std::vector<std::vector<Real>>
  > ShareGhostNodes(std::ifstream& istrm) {
    char line[kLineWidth];
    // send nodes info
    std::map<Int, std::vector<Int>> send_nodes;
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int i_part, m_node;
      std::sscanf(line, "%d %d", &i_part, &m_node);
      send_nodes[i_part].emplace_back(m_node);
    }
    std::vector<MPI_Request> requests;
    std::vector<std::vector<Real>> send_bufs;
    for (auto& [i_part, nodes] : send_nodes) {
      auto& coords = send_bufs.emplace_back();
      for (auto m_node : nodes) {
        auto& info = m_to_node_info_[m_node];
        auto const& coord = GetCoord(info.i_zone, info.i_node);
        coords.emplace_back(coord[0]);
        coords.emplace_back(coord[1]);
        coords.emplace_back(coord[2]);
      }
      assert(std::is_sorted(nodes.begin(), nodes.end()));
      int n_reals = 3 * nodes.size();
      int tag = i_part;
      auto& request = requests.emplace_back();
      MPI_Isend(coords.data(), n_reals, kMpiRealType, i_part, tag,
          MPI_COMM_WORLD, &request);
    }
    // recv nodes info
    std::map<Int, std::vector<Int>> recv_nodes;
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int i_part, m_node, i_zone, i_node;
      std::sscanf(line, "%d %d %d %d", &i_part, &m_node, &i_zone, &i_node);
      recv_nodes[i_part].emplace_back(m_node);
      m_to_node_info_[m_node] = NodeInfo<Int>(i_zone, i_node);
    }
    std::vector<std::vector<Real>> recv_coords;
    for (auto& [i_part, nodes] : recv_nodes) {
      assert(std::is_sorted(nodes.begin(), nodes.end()));
      int n_reals = 3 * nodes.size();
      auto& coords = recv_coords.emplace_back(std::vector<Real>(n_reals));
      int tag = rank_;
      auto& request = requests.emplace_back();
      MPI_Irecv(coords.data(), n_reals, kMpiRealType, i_part, tag,
          MPI_COMM_WORLD, &request);
    }
    // wait until all send/recv finish
    std::vector<MPI_Status> statuses(requests.size());
    MPI_Waitall(requests.size(), requests.data(), statuses.data());
    requests.clear();
    return { recv_nodes, recv_coords };
  }
  void BuildGhostNodes(const std::map<Int, std::vector<Int>>& recv_nodes,
      const std::vector<std::vector<Real>>& recv_coords) {
    // copy node coordinates from buffer to member
    int i_source = 0;
    for (auto& [i_part, nodes] : recv_nodes) {
      auto* xyz = recv_coords[i_source++].data();
      for (auto m_node : nodes) {
        auto& info = m_to_node_info_[m_node];
        ghost_nodes_[info.i_zone][info.i_node] = { xyz[0], xyz[1] , xyz[2] };
        xyz += 3;
      }
    }
  }
  struct Conn {
    ShiftedVector<Int> index;
    std::vector<Int> nodes;
  };
  using ZoneSectToConn = std::map<Int, std::map<Int, Conn>>;
  ZoneSectToConn BuildLocalCells(std::ifstream& istrm, int i_file) {
    char line[kLineWidth];
    // build local cells
    auto cell_conn = ZoneSectToConn();
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int i_zone, i_sect, head, tail;
      std::sscanf(line, "%d %d %d %d", &i_zone, &i_sect, &head, &tail);
      local_cells_[i_zone][i_sect] = CellGroupType(head, tail - head);
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
      if (cgp_field_general_read_data(i_file, i_base, i_zone, 2, 2,
          range_min, range_max, kIntType,
          1, mem_dimensions, mem_range_min, mem_range_max, metis_ids.data()))
        cgp_error_exit();
      for (int i_cell = head; i_cell < tail; ++i_cell) {
        auto m_cell = metis_ids[i_cell];
        m_to_cell_info_[m_cell] = CellInfo<Int>(i_zone, i_sect, i_cell);
      }
      char name[33];
      ElementType_t type;
      cgsize_t u, v;
      int x, y, npe;
      auto& conn = cell_conn[i_zone][i_sect];
      auto& index = conn.index;
      auto& nodes = conn.nodes;
      index = ShiftedVector<Int>(mem_dimensions[0] + 1, head);
      if (cg_section_read(i_file, i_base, i_zone, i_sect, name, &type,
          &u, &v, &x, &y))
        cgp_error_exit();
      cg_npe(type, &npe);
      nodes.resize(npe * mem_dimensions[0]);
      for (int i = 0; i < index.size(); ++i) {
        index.at(head + i) = npe * i;
      }
      if (cgp_elements_read_data(i_file, i_base, i_zone, i_sect,
          range_min[0], range_max[0], nodes.data()))
        cgp_error_exit();
      for (int i_cell = head; i_cell < tail; ++i_cell) {
        auto* i_node_list = &nodes[(i_cell - head) * 8];
        auto hexa_ptr = std::make_unique<integrator::Hexa<Real, 4, 4, 4>>(
            GetCoord(i_zone, i_node_list[0]), GetCoord(i_zone, i_node_list[1]),
            GetCoord(i_zone, i_node_list[2]), GetCoord(i_zone, i_node_list[3]),
            GetCoord(i_zone, i_node_list[4]), GetCoord(i_zone, i_node_list[5]),
            GetCoord(i_zone, i_node_list[6]), GetCoord(i_zone, i_node_list[7]));
        auto cell = CellType(
            std::move(hexa_ptr), metis_ids[i_cell]);
        local_cells_[i_zone][i_sect][i_cell] = std::move(cell);
      }
    }
    return cell_conn;
  }
  void AddLocalCellId() {
    for (auto& [i_zone, zone] : local_cells_) {
      for (auto& [i_sect, sect] : zone) {
        for (auto& cell : sect) {
          if (cell.inner()) {
            inner_cells_.emplace_back(&cell);
          } else {
            inter_cells_.emplace_back(&cell);
          }
        }
      }
    }
    Int id = 0;
    for (auto cell_ptr : inner_cells_) {
      cell_ptr->id_ = id++;
    }
    for (auto cell_ptr : inter_cells_) {
      cell_ptr->id_ = id++;
    }
  }
  struct GhostAdj {
    std::map<Int, std::map<Int, Int>>
        send_node_cnts, recv_node_cnts;  // [i_part][m_cell] -> node_cnt
    std::vector<std::pair<Int, Int>>
        m_cell_pairs;
  };
  GhostAdj BuildAdj(std::ifstream& istrm) {
    char line[kLineWidth];
    // local adjacency
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int i, j;
      std::sscanf(line, "%d %d", &i, &j);
      local_adjs_.emplace_back(i, j);
    }
    // ghost adjacency
    auto ghost_adj = GhostAdj();
    auto& send_node_cnts = ghost_adj.send_node_cnts;
    auto& recv_node_cnts = ghost_adj.recv_node_cnts;
    auto& m_cell_pairs = ghost_adj.m_cell_pairs;
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int p, i, j, cnt_i, cnt_j;
      std::sscanf(line, "%d %d %d %d %d", &p, &i, &j, &cnt_i, &cnt_j);
      send_node_cnts[p][i] = cnt_i;
      recv_node_cnts[p][j] = cnt_j;
      m_cell_pairs.emplace_back(i, j);
    }
    return ghost_adj;
  }
  auto ShareGhostCells(const GhostAdj& ghost_adj,
      const ZoneSectToConn& cell_conn) {
    auto& send_node_cnts = ghost_adj.send_node_cnts;
    auto& recv_node_cnts = ghost_adj.recv_node_cnts;
    // send cell.i_zone and cell.node_id_list
    std::vector<std::vector<Int>> send_cells;
    std::vector<MPI_Request> requests;
    for (auto& [i_part, node_cnts] : send_node_cnts) {
      auto& send_buf = send_cells.emplace_back();
      for (auto& [m_cell, cnt] : node_cnts) {
        auto& info = m_to_cell_info_[m_cell];
        Int i_zone = m_to_cell_info_[m_cell].i_zone,
            i_sect = m_to_cell_info_[m_cell].i_sect,
            i_cell = m_to_cell_info_[m_cell].i_cell;
        auto& conn = cell_conn.at(i_zone).at(i_sect);
        auto& index = conn.index;
        auto& nodes = conn.nodes;
        auto* i_node_list = &(nodes[index[i_cell]]);
        send_buf.emplace_back(i_zone);
        for (int i = 0; i < cnt; ++i) {
          send_buf.emplace_back(i_node_list[i]);
        }
      }
      int n_ints = send_buf.size();
      int tag = i_part;
      auto& request = requests.emplace_back();
      MPI_Isend(send_buf.data(), n_ints, kMpiIntType, i_part, tag,
          MPI_COMM_WORLD, &request);
    }
    // recv cell.i_zone and cell.node_id_list
    std::vector<std::vector<Int>> recv_cells;
    for (auto& [i_part, node_cnts] : recv_node_cnts) {
      auto& recv_buf = recv_cells.emplace_back();
      int n_ints = 0;
      for (auto& [m_cell, cnt] : node_cnts) {
        ++n_ints;
        n_ints += cnt;
      }
      int tag = rank_;
      recv_buf.resize(n_ints);
      auto& request = requests.emplace_back();
      MPI_Irecv(recv_buf.data(), n_ints, kMpiIntType, i_part, tag,
          MPI_COMM_WORLD, &request);
    }
    // wait until all send/recv finish
    std::vector<MPI_Status> statuses(requests.size());
    MPI_Waitall(requests.size(), requests.data(), statuses.data());
    requests.clear();
    return recv_cells;
  }
  std::unordered_map<Int, std::pair<int, int>> BuildGhostCells(
      const GhostAdj& ghost_adj,
      const std::vector<std::vector<Int>>& recv_cells) {
    auto& recv_node_cnts = ghost_adj.recv_node_cnts;
    // build ghost cells
    std::unordered_map<Int, std::pair<int, int>> m_to_recv_cells;
    int i_source = 0;
    for (auto& [i_part, node_cnts] : recv_node_cnts) {
      auto& recv_buf = recv_cells.at(i_source);
      int index = 0;
      for (auto& [m_cell, cnt] : node_cnts) {
        m_to_recv_cells[m_cell].first = i_source;
        m_to_recv_cells[m_cell].second = index + 1;
        int i_zone = recv_buf[index++];
        auto* i_node_list = &recv_buf[index];
        auto hexa_ptr = std::make_unique<integrator::Hexa<Real, 4, 4, 4>>(
          GetCoord(i_zone, i_node_list[0]), GetCoord(i_zone, i_node_list[1]),
          GetCoord(i_zone, i_node_list[2]), GetCoord(i_zone, i_node_list[3]),
          GetCoord(i_zone, i_node_list[4]), GetCoord(i_zone, i_node_list[5]),
          GetCoord(i_zone, i_node_list[6]), GetCoord(i_zone, i_node_list[7]));
        ghost_cells_[m_cell] = CellType(
            std::move(hexa_ptr), m_cell);
        index += cnt;
      }
      ++i_source;
    }
    return m_to_recv_cells;
  }
  void FillCellPtrs(const GhostAdj& ghost_adj) {
    // fill `send_cell_ptrs_`
    for (auto& [i_part, node_cnts] : ghost_adj.send_node_cnts) {
      auto& curr_part = send_cell_ptrs_[i_part];
      for (auto& [m_cell, cnt] : node_cnts) {
        auto& info = m_to_cell_info_[m_cell];
        auto& cell = local_cells_.at(info.i_zone).at(info.i_sect)[info.i_cell];
        cell.inner_ = false;
        curr_part.emplace_back(&cell);
      }
      send_coeffs_.emplace_back(node_cnts.size() * kFields);
    }
    // fill `recv_cell_ptrs_`
    for (auto& [i_part, node_cnts] : ghost_adj.recv_node_cnts) {
      auto& curr_part = recv_cell_ptrs_[i_part];
      for (auto& [m_cell, cnt] : node_cnts) {
        auto& cell = ghost_cells_.at(m_cell);
        curr_part.emplace_back(&cell);
      }
      recv_coeffs_.emplace_back(node_cnts.size() * kFields);
    }
    assert(send_cell_ptrs_.size() == send_coeffs_.size());
    assert(recv_cell_ptrs_.size() == recv_coeffs_.size());
    requests_.resize(send_coeffs_.size() + recv_coeffs_.size());
  }
  void BuildLocalFaces(const ZoneSectToConn& cell_conn) {
    // build local faces
    for (auto [m_holder, m_sharer] : local_adjs_) {
      auto& holder_info = m_to_cell_info_[m_holder];
      auto& sharer_info = m_to_cell_info_[m_sharer];
      auto i_zone = holder_info.i_zone;
      // find the common nodes of the holder and the sharer
      auto i_node_cnt = std::unordered_map<Int, Int>();
      auto& conn_i_zone = cell_conn.at(i_zone);
      auto& holder_conn = conn_i_zone.at(holder_info.i_sect);
      auto& sharer_conn = conn_i_zone.at(sharer_info.i_sect);
      auto& holder_nodes = holder_conn.nodes;
      auto& sharer_nodes = sharer_conn.nodes;
      auto holder_head = holder_conn.index[holder_info.i_cell];
      auto sharer_head = sharer_conn.index[sharer_info.i_cell];
      for (int i = 0; i < 8; ++i) {
        ++i_node_cnt[holder_nodes[holder_head + i]];
        ++i_node_cnt[sharer_nodes[sharer_head + i]];
      }
      auto common_nodes = std::vector<Int>();
      common_nodes.reserve(4);
      for (auto [i_node, cnt] : i_node_cnt)
        if (cnt == 2)
          common_nodes.emplace_back(i_node);
      assert(common_nodes.size() == 4);
      // let the normal vector point from holder to sharer
      // see http://cgns.github.io/CGNS_docs_current/sids/conv.figs/hexa_8.png
      auto& zone = local_cells_[i_zone];
      auto& holder = zone[holder_info.i_sect][holder_info.i_cell];
      auto& sharer = zone[sharer_info.i_sect][sharer_info.i_cell];
      integrator::Hexa<Real, 4, 4, 4>::SortNodesOnFace(
          4, &holder_nodes[holder_head], common_nodes.data());
      // build the quad integrator
      auto quad_ptr = std::make_unique<integrator::Quad<Real, kDim, 4, 4>>(
          GetCoord(i_zone, common_nodes[0]), GetCoord(i_zone, common_nodes[1]),
          GetCoord(i_zone, common_nodes[2]), GetCoord(i_zone, common_nodes[3]));
      quad_ptr->BuildNormalFrames();
      auto id = local_faces_.size();
      local_faces_.emplace_back(std::move(quad_ptr), &holder, &sharer, id);
      holder.adj_cells_.emplace_back(&sharer);
      sharer.adj_cells_.emplace_back(&holder);
      holder.adj_faces_.emplace_back(&(local_faces_.back()));
      sharer.adj_faces_.emplace_back(&(local_faces_.back()));
      // rotate riemann solvers
      local_faces_.back().RotateRiemanns();
    }
  }
  void BuildGhostFaces(const GhostAdj& ghost_adj,
      const ZoneSectToConn& cell_conn,
      const std::vector<std::vector<Int>>& recv_cells,
      const std::unordered_map<Int, std::pair<int, int>>& m_to_recv_cells) {
    auto& m_cell_pairs = ghost_adj.m_cell_pairs;
    // build ghost faces
    for (auto [m_holder, m_sharer] : m_cell_pairs) {
      auto& holder_info = m_to_cell_info_[m_holder];
      auto& sharer_info = m_to_recv_cells.at(m_sharer);
      auto i_zone = holder_info.i_zone;
      // find the common nodes of the holder and the sharer
      auto i_node_cnt = std::unordered_map<Int, Int>();
      auto& holder_conn = cell_conn.at(i_zone).at(holder_info.i_sect);
      auto& holder_nodes = holder_conn.nodes;
      auto& sharer_nodes = recv_cells[sharer_info.first];
      auto holder_head = holder_conn.index[holder_info.i_cell];
      auto sharer_head = sharer_info.second;
      for (int i = 0; i < 8; ++i) {
        ++i_node_cnt[holder_nodes[holder_head + i]];
        ++i_node_cnt[sharer_nodes[sharer_head + i]];
      }
      auto common_nodes = std::vector<Int>();
      common_nodes.reserve(4);
      for (auto [i_node, cnt] : i_node_cnt) {
        if (cnt == 2)
          common_nodes.emplace_back(i_node);
      }
      assert(common_nodes.size() == 4);
      // let the normal vector point from holder to sharer
      // see http://cgns.github.io/CGNS_docs_current/sids/conv.figs/hexa_8.png
      auto& zone = local_cells_[i_zone];
      auto& holder = zone[holder_info.i_sect][holder_info.i_cell];
      auto& sharer = ghost_cells_.at(m_sharer);
      integrator::Hexa<Real, 4, 4, 4>::SortNodesOnFace(
          4, &holder_nodes[holder_head], common_nodes.data());
      // build the quad integrator
      auto quad_ptr = std::make_unique<integrator::Quad<Real, kDim, 4, 4>>(
          GetCoord(i_zone, common_nodes[0]), GetCoord(i_zone, common_nodes[1]),
          GetCoord(i_zone, common_nodes[2]), GetCoord(i_zone, common_nodes[3]));
      quad_ptr->BuildNormalFrames();
      auto id = local_faces_.size() + ghost_faces_.size();
      ghost_faces_.emplace_back(std::move(quad_ptr), &holder, &sharer, id);
      holder.adj_cells_.emplace_back(&sharer);
      holder.adj_faces_.emplace_back(&(ghost_faces_.back()));
      // rotate riemann solvers
      ghost_faces_.back().RotateRiemanns();
    }
  }

 public:
  template <class Callable>
  void Project(Callable&& new_func) {
    for (auto& [i_zone, sects] : local_cells_) {
      for (auto& [i_sect, cells] : sects) {
        for (auto& cell : cells) {
          cell.Project(new_func);
        }
      }
    }
  }
  Int CountLocalCells() const {
    return inner_cells_.size() + inter_cells_.size();
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
  void WriteSolutions(const std::string &soln_name = "0") const {
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
      if (cg_sol_write(i_file, i_base, i_zone, soln_name.c_str(),
          CellCenter, &i_soln))
        cgp_error_exit();
      int n_fields = kFields;
      for (int i_field = 1; i_field <= n_fields; ++i_field) {
        int n_sects = zone.size();
        for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
          auto& section = zone.at(i_sect);
          auto field_name = "Field" + std::to_string(i_field);
          int field_id;
          if (cgp_field_write(i_file, i_base, i_zone, i_soln, kRealType,
              field_name.c_str(),  &field_id))
            cgp_error_exit();
          assert(field_id == i_field);
          cgsize_t first[] = { section.head() };
          cgsize_t last[] = { section.tail() - 1 };
          if (cgp_field_write_data(i_file, i_base, i_zone, i_soln, i_field,
              first, last, section.GetField(i_field).data()))
            cgp_error_exit();
        }
      }
    }
    if (cgp_close(i_file))
      cgp_error_exit();
  }
  void WriteSolutionsOnGaussPoints(const std::string &soln_name = "0") const {
    auto ostrm = GetFstream(soln_name);
    ostrm << "# vtk DataFile Version 3.0\n";
    ostrm << "Field values on quadrature points.\n";
    ostrm << "ASCII\n";
    ostrm << "DATASET UNSTRUCTURED_GRID\n";
    int n_points = 0;
    auto coords = std::vector<Mat3x1>();
    auto fields = std::vector<typename CellType::Value>();
    for (auto& [i_zone, zone] : local_cells_) {
      for (auto& [i_sect, sect] : zone) {
        for (auto& cell : sect) {
          auto& gauss_ptr = cell.gauss_ptr_;
          for (int q = 0; q < gauss_ptr->CountQuadPoints(); ++q) {
            coords.emplace_back(gauss_ptr->GetGlobalCoord(q));
            fields.emplace_back(cell.projection_(coords.back()));
          }
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
      ostrm << "LOOKUP_TABLE default\n";
      for (auto& f : fields) {
        ostrm << f[k] << '\n';
      }
    }
  }
  void WriteSolutionsOnCellCenters(const std::string &soln_name = "0") const {
    bool binary = false;
    auto ostrm = GetFstream(soln_name);
    ostrm << "# vtk DataFile Version 3.0\n";
    ostrm << "Field values on each cell.\n";
    ostrm << (binary ? "BINARY\n" : "ASCII\n");
    ostrm << "DATASET UNSTRUCTURED_GRID\n";
    int n_points = 0;
    auto coords = std::vector<Mat3x1>();
    auto cells = std::vector<Int>();
    auto fields = std::vector<typename CellType::Value>();
    for (auto& [i_zone, zone] : local_cells_) {
      for (auto& [i_sect, sect] : zone) {
        int i_cell = sect.head();
        int i_cell_tail = sect.tail();
        while (i_cell < i_cell_tail) {
          cells.emplace_back(8);
          auto& gauss_ptr = sect[i_cell].gauss_ptr_;
          auto& proj = sect[i_cell].projection_;
          // nodes at corners
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({-1, -1, -1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({+1, -1, -1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({+1, +1, -1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({-1, +1, -1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({-1, -1, +1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({+1, -1, +1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({+1, +1, +1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({-1, +1, +1}));
          fields.emplace_back(proj(coords.back()));
          /* nodes on edges
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({0., -1, -1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({+1, 0., -1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({0., +1, -1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({-1, 0., -1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({0., -1, +1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({+1, 0., +1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({0., +1, +1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({-1, 0., +1}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({-1, -1, 0.}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({+1, -1, 0.}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({+1, +1, 0.}));
          fields.emplace_back(proj(coords.back()));
          cells.emplace_back(coords.size());
          coords.emplace_back(gauss_ptr->LocalToGlobal({-1, +1, 0.}));
          fields.emplace_back(proj(coords.back())); */
          ++i_cell;
        }
      }
    }
    ostrm << "POINTS " << coords.size() << " double\n";
    for (auto& xyz : coords) {
      if (binary) {
        ostrm.write(reinterpret_cast<char*>(&xyz[0]), sizeof(xyz[0]));
        ostrm.write(reinterpret_cast<char*>(&xyz[1]), sizeof(xyz[1]));
        ostrm.write(reinterpret_cast<char*>(&xyz[2]), sizeof(xyz[2]));
      } else {
        ostrm << xyz[0] << ' ' << xyz[1] << ' ' << xyz[2] << '\n';
      }
    }
    ostrm << '\n';
    auto n_cells = cells.size() / (8 + 1);
    ostrm << "CELLS " << n_cells << ' ' << cells.size() << '\n';
    for (auto& c : cells) {
      if (binary) {
        Int ic = c;
        ostrm.write(reinterpret_cast<char*>(&ic), sizeof(ic));
      } else {
        ostrm << c << ' ';
      }
    }
    ostrm << '\n';
    ostrm << "CELL_TYPES " << n_cells << '\n';
    Int t = 12;  // VTK_HEXAHEDRON
    // Int type = 25;  // VTK_QUADRATIC_HEXAHEDRON
    for (int i = 0; i < n_cells; ++i) {
      if (binary) {
        ostrm.write(reinterpret_cast<char*>(&t), sizeof(t));
      } else {
        ostrm << t << ' ';
      }
    }
    ostrm << '\n';
    ostrm << "POINT_DATA " << coords.size() << "\n";
    int K = fields[0].size();
    for (int k = 0; k < K; ++k) {
      ostrm << "SCALARS Field[" << k + 1 << "] double 1\n";
      ostrm << "LOOKUP_TABLE default\n";
      for (auto& f : fields) {
        if (binary) {
          ostrm.write(reinterpret_cast<char*>(&f[k]), sizeof(f[k]));
        } else {
          ostrm << f[k] << ' ';
        }
      }
      ostrm << '\n';
    }
  }
  void ShareGhostCellCoeffs() {
    int i_req = 0;
    // send cell.projection_.coeff_
    int i_buf = 0;
    for (auto& [i_part, cell_ptrs] : send_cell_ptrs_) {
      auto& send_buf = send_coeffs_[i_buf++];
      int i_real = 0;
      for (auto* cell_ptr : cell_ptrs) {
        auto& coeff = cell_ptr->projection_.coeff();
        const auto& coeff_vec_view = coeff.reshaped();
        for (int i = 0; i < kFields; ++i) {
          send_buf[i_real++] = coeff_vec_view[i];
        }
      }
      int tag = i_part;
      auto& request = requests_[i_req++];
      MPI_Isend(send_buf.data(), send_buf.size(), kMpiRealType, i_part, tag,
          MPI_COMM_WORLD, &request);
    }
    // recv cell.projection_.coeff_
    i_buf = 0;
    for (auto& [i_part, cell_ptrs] : recv_cell_ptrs_) {
      auto& recv_buf = recv_coeffs_[i_buf++];
      int tag = rank_;
      auto& request = requests_[i_req++];
      MPI_Irecv(recv_buf.data(), recv_buf.size(), kMpiRealType, i_part, tag,
          MPI_COMM_WORLD, &request);
    }
    assert(i_req == send_coeffs_.size() + recv_coeffs_.size());
  }
  void UpdateGhostCellCoeffs() {
    // wait until all send/recv finish
    std::vector<MPI_Status> statuses(requests_.size());
    MPI_Waitall(requests_.size(), requests_.data(), statuses.data());
    int req_size = requests_.size();
    requests_.clear();
    requests_.resize(req_size);
    // update coeffs
    int i_buf = 0;
    for (auto& [i_part, cell_ptrs] : recv_cell_ptrs_) {
      auto* recv_buf = recv_coeffs_[i_buf++].data();
      for (auto* cell_ptr : cell_ptrs) {
        cell_ptr->projection_.UpdateCoeffs(recv_buf);
        recv_buf += kFields;
      }
    }
  }

  template <typename Callable>
  void Reconstruct(Callable&& limiter) {
    ShareGhostCellCoeffs();
    // run the limiter on inner cells that need no ghost cells
    for (auto* cell_ptr : inner_cells_) {
      limiter(cell_ptr);
    }
    // run the limiter on inter cells that need ghost cells
    UpdateGhostCellCoeffs();
    for (auto* cell_ptr : inter_cells_) {
      limiter(cell_ptr);
    }
  }

  template<class Visitor>
  void ForEachLocalCell(Visitor&& visit) const {
    for (auto& [i_zone, zone] : local_cells_) {
      for (auto& [i_sect, sect] : zone) {
        for (auto& cell : sect) {
          visit(cell);
        }
      }
    }
  }
  template<class Visitor>
  void ForEachLocalCell(Visitor&& visit) {
    for (auto& [i_zone, zone] : local_cells_) {
      for (auto& [i_sect, sect] : zone) {
        for (auto& cell : sect) {
          // TODO(PVC): pass ptr to mutators
          visit(cell);
        }
      }
    }
  }
  template<class Visitor>
  void ForEachLocalFace(Visitor&& visit) {
    for (auto& face : local_faces_) {
      visit(face);
    }
  }
  template<class Visitor>
  void ForEachGhostFace(Visitor&& visit) {
    for (auto& face : ghost_faces_) {
      visit(face);
    }
  }
  template<class Visitor>
  void ForEachSolidFace(Visitor&& visit) {
    for (auto& [i_zone, zone] : bound_faces_) {
      for (auto& [i_sect, sect] : zone) {
        for (auto& face : sect) {
          visit(face);
        }
      }
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
  std::map<Int, std::map<Int, CellGroupType>>
      local_cells_;  // [i_zone][i_sect][i_cell] -> a Cell obj
  std::vector<CellPtr>
      inner_cells_, inter_cells_;  // [i_cell] -> CellPtr
  std::map<Int, std::vector<CellPtr>>
      send_cell_ptrs_, recv_cell_ptrs_;  // [i_part] -> vector<CellPtr>
  std::vector<std::vector<Real>>
      send_coeffs_, recv_coeffs_;
  std::unordered_map<Int, CellType>
      ghost_cells_;  //                 [m_cell] -> a Cell obj
  std::vector<std::pair<Int, Int>>
      local_adjs_;  // [i_pair] -> { m_holder, m_sharer }
  std::vector<FaceType>
      local_faces_, ghost_faces_;  // [i_face] -> a Face obj
  std::map<Int, std::map<Int, ShiftedVector<FaceType>>>
      bound_faces_;  // [i_zone][i_sect][i_face] -> a Face obj
  std::vector<MPI_Request> requests_;
  const std::string directory_;
  const std::string cgns_file_;
  const std::string part_path_;
  int rank_;

  void BuildBoundaryFaces(const ZoneSectToConn &cell_conn,
      std::ifstream& istrm, int i_file) {
    std::unordered_map<Int, std::unordered_map<Int, std::vector<Int>>>
        z_n_to_m_cells;  // [i_zone][i_node] -> vector of `m_cell`s
    for (auto& [i_zone, zone] : local_cells_) {
      auto& n_to_m_cells = z_n_to_m_cells[i_zone];
      for (auto& [i_sect, sect] : zone) {
        auto& conn = cell_conn.at(i_zone).at(i_sect);
        auto& index = conn.index;
        auto& nodes = conn.nodes;
        for (int i_cell = sect.head(); i_cell < sect.tail(); ++i_cell) {
          auto& cell = sect[i_cell];
          auto m_cell = cell.metis_id;
          for (int i = index.at(i_cell); i < index.at(i_cell+1); ++i) {
            n_to_m_cells[nodes[i]].emplace_back(m_cell);
          }
        }
      }
    }
    char line[kLineWidth];
    auto face_conn = ZoneSectToConn();
    // build local faces
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int i_zone, i_sect, head, tail;
      std::sscanf(line, "%d %d %d %d", &i_zone, &i_sect, &head, &tail);
      auto& faces = bound_faces_[i_zone][i_sect];
      cgsize_t range_min[] = { head };
      cgsize_t range_max[] = { tail - 1 };
      cgsize_t mem_dimensions[] = { tail - head };
      char name[33];
      ElementType_t type;
      cgsize_t u, v;
      int x, y, npe;
      auto& conn = face_conn[i_zone][i_sect];
      auto& index = conn.index;
      auto& nodes = conn.nodes;
      if (cg_section_read(i_file, i_base, i_zone, i_sect, name, &type,
          &u, &v, &x, &y))
        cgp_error_exit();
      cg_npe(type, &npe);
      nodes.resize(npe * mem_dimensions[0]);
      nodes = std::vector<Int>(npe * mem_dimensions[0]);
      index = ShiftedVector<Int>(mem_dimensions[0] + 1, head);
      for (int i = 0; i < index.size(); ++i) {
        index.at(head + i) = npe * i;
      }
      if (cgp_elements_read_data(i_file, i_base, i_zone, i_sect,
          range_min[0], range_max[0], nodes.data()))
        cgp_error_exit();
      auto& n_to_m_cells = z_n_to_m_cells.at(i_zone);
      for (int i_face = head; i_face < tail; ++i_face) {
        auto* i_node_list = &nodes[(i_face - head) * npe];
        auto cell_cnt = std::unordered_map<int, int>();
        for (int i = index.at(i_face); i < index.at(i_face+1); ++i) {
          for (auto m_cell : n_to_m_cells[nodes[i]]) {
            cell_cnt[m_cell]++;
          }
        }
        CellType *holder_ptr;
        for (auto [m_cell, cnt] : cell_cnt) {
          assert(cnt <= npe);
          if (cnt == npe) {
            auto& info = m_to_cell_info_[m_cell];
            Int z = info.i_zone, s = info.i_sect, c = info.i_cell;
            holder_ptr = &(local_cells_.at(z).at(s).at(c));
            auto& holder_conn = cell_conn.at(z).at(s);
            auto& holder_nodes = holder_conn.nodes;
            auto holder_head = holder_conn.index[c];
            integrator::Hexa<Real, 4, 4, 4>::SortNodesOnFace(
                npe, &holder_nodes[holder_head], i_node_list);
            break;
          }
        }
        auto quad_ptr = std::make_unique<integrator::Quad<Real, kDim, 4, 4>>(
            GetCoord(i_zone, i_node_list[0]), GetCoord(i_zone, i_node_list[1]),
            GetCoord(i_zone, i_node_list[2]), GetCoord(i_zone, i_node_list[3]));
        quad_ptr->BuildNormalFrames();
        auto face = FaceType(std::move(quad_ptr), holder_ptr, nullptr);
        faces.emplace_back(std::move(face));
        faces.back().RotateRiemanns();
        holder_ptr->adj_faces_.emplace_back(&faces.back());
      }
    }
    // link boundary faces to cells
    // for (auto& [i_zone, zone] : local_cells_) {
      // [i_node] -> vector of `m_cell`s
      // auto node_users = std::unordered_map<Int, std::vector<Int>>();
      // for (auto& [i_sect, sect] : zone) {
      //   auto& conn = cell_conn.at(i_zone).at(i_sect);
      //   auto& index = conn.index;
      //   auto& nodes = conn.nodes;
      //   for (int i_cell = sect.head(); i_cell < sect.tail(); ++i_cell) {
      //     auto& cell = sect[i_cell];
      //     auto m_cell = cell.metis_id;
      //     for (int i = index.at(i_cell); i < index.at(i_cell+1); ++i) {
      //       node_users[nodes[i]].emplace_back(m_cell);
      //     }
      //   }
      // }
      // auto& face_zone = face_conn.at(i_zone);
      // for (auto& [i_sect, sect] : face_zone) {
      //   auto& conn = face_zone.at(i_sect);
      //   auto& index = conn.index;
      //   auto& nodes = conn.nodes;
      //   int n_faces =  index.size() - 1;
      //   for (int i_face = 0; i_face < n_faces; ++i_face) {
      //   }
      // }
    // }
  }

  Mat3x1 GetCoord(int i_zone, int i_node) const {
    Mat3x1 coord;
    auto iter_zone = local_nodes_.find(i_zone);
    if (iter_zone != local_nodes_.end() && iter_zone->second.has(i_node)) {
      coord[0] = iter_zone->second.x_[i_node];
      coord[1] = iter_zone->second.y_[i_node];
      coord[2] = iter_zone->second.z_[i_node];
    } else {
      coord = ghost_nodes_.at(i_zone).at(i_node);
    }
    return coord;
  }
  std::ofstream GetFstream(const std::string &soln_name) const {
    char temp[1024];
    if (rank_ == 0) {
      std::snprintf(temp, sizeof(temp), "mkdir -p %s/%s",
          directory_.c_str(), soln_name.c_str());
      std::system(temp);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::snprintf(temp, sizeof(temp), "%s/%s/%d.vtk",
        directory_.c_str(), soln_name.c_str(), rank_);
    return std::ofstream(temp);
  }
};
template <typename Int, typename Real, int kFunc, int kDim, int kOrder>
MPI_Datatype const Part<Int, Real, kFunc, kDim, kOrder>::kMpiIntType
    = sizeof(Int) == 8 ? MPI_LONG : MPI_INT;
template <typename Int, typename Real, int kFunc, int kDim, int kOrder>
MPI_Datatype const Part<Int, Real, kFunc, kDim, kOrder>::kMpiRealType
    = sizeof(Real) == 8 ? MPI_DOUBLE : MPI_FLOAT;

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_PART_HPP_
