// Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_MESH_PART_HPP_
#define MINI_MESH_PART_HPP_

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <ios>
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
#include "mini/integrator/tri.hpp"
#include "mini/integrator/quad.hpp"
#include "mini/integrator/tetra.hpp"
#include "mini/integrator/hexa.hpp"
#include "mini/polynomial/basis.hpp"
#include "mini/polynomial/projection.hpp"

namespace mini {
namespace mesh {
namespace cgns {

template <class Object>
void print(Object&& obj) {
  std::cout << obj << '\n' << std::endl;
}

template <class Int>
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
template <class Int>
struct CellInfo {
  CellInfo() = default;
  CellInfo(Int z, Int s, Int c, Int n)
      : i_zone(z), i_sect(s), i_cell(c), npe(n) {}
  CellInfo(CellInfo const&) = default;
  CellInfo& operator=(CellInfo const&) = default;
  CellInfo(CellInfo&&) noexcept = default;
  CellInfo& operator=(CellInfo&&) noexcept = default;
  ~CellInfo() noexcept = default;
  Int i_zone{0}, i_sect{0}, i_cell{0}, npe{0};
};

template <typename Int, typename Scalar>
struct NodeGroup {
  Int head_, size_;
  ShiftedVector<Int> metis_id_;
  ShiftedVector<Scalar> x_, y_, z_;

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

template <typename Int, int kOrder, class R>
struct Cell;

template <typename Int, int kOrder, class R>
struct Face {
  using Riemann = R;
  using Scalar = typename Riemann::Scalar;
  constexpr static int kFunc = Riemann::kFunc;
  constexpr static int kDim = Riemann::kDim;
  using Gauss = integrator::Face<Scalar, kDim>;
  using GaussUptr = std::unique_ptr<Gauss>;
  using Cell = cgns::Cell<Int, kOrder, Riemann>;

  GaussUptr gauss_ptr_;
  Cell *holder_, *sharer_;
  Riemann riemann_;
  Int id_{-1};

  Face(GaussUptr&& gauss_ptr, Cell *holder, Cell *sharer, Int id = 0)
      : gauss_ptr_(std::move(gauss_ptr)), holder_(holder), sharer_(sharer),
        id_(id) {
    riemann_.Rotate(gauss_ptr_->GetNormalFrame(0));
  }
  Face(const Face&) = delete;
  Face& operator=(const Face&) = delete;
  Face(Face&&) noexcept = default;
  Face& operator=(Face&&) noexcept = default;
  ~Face() noexcept = default;

  const Gauss& gauss() const {
    return *gauss_ptr_;
  }
  Scalar area() const {
    return gauss_ptr_->area();
  }
  Int id() const {
    return id_;
  }
  const Cell *other(const Cell *cell) const {
    assert(cell == sharer_ || cell == holder_);
    return cell == holder_ ? sharer_ : holder_;
  }
};

template <typename Int, int kOrder, class R>
struct Cell {
  using Riemann = R;
  using Scalar = typename Riemann::Scalar;
  constexpr static int kFunc = Riemann::kFunc;
  constexpr static int kDim = Riemann::kDim;
  using Gauss = integrator::Cell<Scalar>;
  using GaussUptr = std::unique_ptr<Gauss>;
  using Basis = polynomial::OrthoNormal<Scalar, kDim, kOrder>;
  using Projection = polynomial::Projection<Scalar, kDim, kOrder, kFunc>;
  using Coord = typename Projection::Coord;
  using Value = typename Projection::Value;
  using Coeff = typename Projection::Coeff;
  static constexpr int K = Projection::K;  // number of functions
  static constexpr int N = Projection::N;  // size of the basis
  static constexpr int kFields = K * N;
  using Face = cgns::Face<Int, kOrder, R>;

  std::vector<Cell*> adj_cells_;
  std::vector<Face*> adj_faces_;
  Basis basis_;
  GaussUptr gauss_ptr_;
  Projection projection_;
  Int metis_id{-1}, id_{-1};
  bool inner_ = true;

  Cell(GaussUptr&& gauss_ptr, Int m_cell)
      : basis_(*gauss_ptr), gauss_ptr_(std::move(gauss_ptr)),
        metis_id(m_cell), projection_(basis_) {
  }
  Cell() = default;
  Cell(const Cell&) = delete;
  Cell& operator=(const Cell&) = delete;
  Cell(Cell&& that) noexcept {
    *this = std::move(that);
  }
  Cell& operator=(Cell&& that) noexcept {
    adj_cells_ = std::move(that.adj_cells_);
    adj_faces_ = std::move(that.adj_faces_);
    basis_ = std::move(that.basis_);
    gauss_ptr_ = std::move(that.gauss_ptr_);
    projection_ = std::move(that.projection_);
    projection_.basis_ptr_ = &basis_;  //
    metis_id = that.metis_id;
    id_ = that.id_;
    inner_ = that.inner_;
    return *this;
  }
  ~Cell() noexcept = default;

  Scalar volume() const {
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
  const Gauss& gauss() const {
    return *gauss_ptr_;
  }
  static int GetVtkType(int n_corners) {
    int type;
    switch (n_corners) {
      case 4:
        type = 10;  // VTK_TETRA
        break;
      case 8:
        // type = 12;  // VTK_HEXAHEDRON
        type = 25;  // VTK_QUADRATIC_HEXAHEDRON
        break;
      default:
        assert(false);
        break;
    }
    return type;
  }
  static int CountNodes(int vtk_type) {
    int n_nodes;
    switch (vtk_type) {
      case 10:  // VTK_TETRA
        n_nodes = 4;
        break;
      case 12:  // VTK_HEXAHEDRON
        n_nodes = 8;
        break;
      case 25:  // VTK_QUADRATIC_HEXAHEDRON
        n_nodes = 20;
        break;
      default:
        assert(false);
        break;
    }
    return n_nodes;
  }
  void WriteSolutions(int vtk_type, std::vector<Coord> *coords,
      std::vector<Value> *fields) const {
    switch (vtk_type) {
      case 10:  // VTK_TETRA
        WriteSolutionsOnTetra4(coords, fields);
        break;
      case 12:  // VTK_HEXAHEDRON
        WriteSolutionsOnHexa8(coords, fields);
        break;
      case 25:  // VTK_QUADRATIC_HEXAHEDRON
        WriteSolutionsOnHexa20(coords, fields);
        break;
      default:
        assert(false);
        break;
    }
  }

  template <class Callable>
  void Project(Callable&& func) {
    projection_.Project(func, basis_);
  }

 private:
  void WriteSolutionsOnTetra4(std::vector<Coord> *coords,
      std::vector<Value> *fields) const {
    coords->emplace_back(gauss().LocalToGlobal({1, 0, 0}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({0, 1, 0}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({0, 0, 1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({0, 0, 0}));
    fields->emplace_back(projection_(coords->back()));
  }
  void WriteSolutionsOnHexa8(std::vector<Coord> *coords,
      std::vector<Value> *fields) const {
    coords->emplace_back(gauss().LocalToGlobal({-1, -1, -1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({+1, -1, -1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({+1, +1, -1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({-1, +1, -1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({-1, -1, +1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({+1, -1, +1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({+1, +1, +1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({-1, +1, +1}));
    fields->emplace_back(projection_(coords->back()));
  }
  void WriteSolutionsOnHexa20(std::vector<Coord> *coords,
      std::vector<Value> *fields) const {
    // nodes at corners
    WriteSolutionsOnHexa8(coords, fields);
    // nodes on edges
    coords->emplace_back(gauss().LocalToGlobal({0., -1, -1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({+1, 0., -1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({0., +1, -1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({-1, 0., -1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({0., -1, +1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({+1, 0., +1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({0., +1, +1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({-1, 0., +1}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({-1, -1, 0.}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({+1, -1, 0.}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({+1, +1, 0.}));
    fields->emplace_back(projection_(coords->back()));
    coords->emplace_back(gauss().LocalToGlobal({-1, +1, 0.}));
    fields->emplace_back(projection_(coords->back()));
  }
};

template <typename Int, int kOrder, class R>
class CellGroup {
  using Cell = cgns::Cell<Int, kOrder, R>;
  using Scalar = typename Cell::Scalar;

  Int head_, size_;
  ShiftedVector<Cell> cells_;
  ShiftedVector<ShiftedVector<Scalar>> fields_;  // [i_field][i_cell]
  int npe_;

 public:
  static constexpr int kFields = Cell::kFields;

  CellGroup(int head, int size, int npe)
      : head_(head), size_(size), cells_(size, head), fields_(kFields, 1),
        npe_(npe) {
    for (int i = 1; i <= kFields; ++i) {
      fields_[i] = ShiftedVector<Scalar>(size, head);
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
  int npe() const {
    return npe_;
  }
  bool has(int i_cell) const {
    return head() <= i_cell && i_cell < tail();
  }
  const Cell& operator[](Int i_cell) const {
    return cells_[i_cell];
  }
  Cell& operator[](Int i_cell) {
    return cells_[i_cell];
  }
  const Cell& at(Int i_cell) const {
    return cells_.at(i_cell);
  }
  Cell& at(Int i_cell) {
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
  const ShiftedVector<Scalar>& GetField(Int i_field) const {
    return fields_[i_field];
  }
  ShiftedVector<Scalar>& GetField(Int i_field) {
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
  void ScatterFields() {
    for (int i_cell = head(); i_cell < tail(); ++i_cell) {
      auto& cell = cells_.at(i_cell);
      auto& coeff = cell.projection_.coeff();
      for (int i_field = 1; i_field <= kFields; ++i_field) {
        coeff.reshaped()[i_field-1] = fields_.at(i_field).at(i_cell);
      }
    }
  }
};

template <typename Int, int kOrder, class R>
class Part {
 public:
  using Face = cgns::Face<Int, kOrder, R>;
  using Cell = cgns::Cell<Int, kOrder, R>;
  using Riemann = R;
  using Scalar = typename Riemann::Scalar;
  constexpr static int kFunc = Riemann::kFunc;
  constexpr static int kDim = Riemann::kDim;

 private:
  using Mat3x1 = algebra::Matrix<Scalar, 3, 1>;
  using NodeInfo = cgns::NodeInfo<Int>;
  using CellInfo = cgns::CellInfo<Int>;
  using NodeGroup = cgns::NodeGroup<Int, Scalar>;
  using CellGroup = cgns::CellGroup<Int, kOrder, R>;
  static constexpr int kLineWidth = 128;
  static constexpr int kFields = CellGroup::kFields;
  static constexpr int i_base = 1;
  static constexpr auto kIntType
      = sizeof(Int) == 8 ? CGNS_ENUMV(LongInteger) : CGNS_ENUMV(Integer);
  static constexpr auto kRealType
      = sizeof(Scalar) == 8 ? CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
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
  void SetFieldNames(const std::array<std::string, kFunc>& names) {
    field_names_ = names;
  }

 private:
  int SolnNameToId(int i_file, int i_base, int i_zone,
      const std::string &name) {
    int n_solns;
    if (cg_nsols(i_file, i_base, i_zone, &n_solns))
      cgp_error_exit();
    int i_soln;
    for (i_soln = 1; i_soln <= n_solns; ++i_soln) {
      char soln_name[33];
      GridLocation_t loc;
      if (cg_sol_info(i_file, i_base, i_zone, i_soln, soln_name, &loc))
        cgp_error_exit();
      if (soln_name == name)
        break;
    }
    assert(i_soln <= n_solns);
    return i_soln;
  }
  int FieldNameToId(int i_file, int i_base, int i_zone, int i_soln,
      const std::string &name) {
    int n_fields;
    if (cg_nfields(i_file, i_base, i_zone, i_soln, &n_fields))
      cgp_error_exit();
    int i_field;
    for (i_field = 1; i_field <= n_fields; ++i_field) {
      char field_name[33];
      DataType_t data_t;
      if (cg_field_info(i_file, i_base, i_zone, i_soln, i_field,
          &data_t, field_name))
        cgp_error_exit();
      if (field_name == name)
        break;
    }
    assert(i_field <= n_fields);
    return i_field;
  }
  void BuildLocalNodes(std::ifstream& istrm, int i_file) {
    char line[kLineWidth];
    istrm.getline(line, kLineWidth); assert(line[0] == '#');
    // node coordinates
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int i_zone, head, tail;
      std::sscanf(line, "%d %d %d", &i_zone, &head, &tail);
      local_nodes_[i_zone] = NodeGroup(head, tail - head);
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
      int i_sol = SolnNameToId(i_file, i_base, i_zone, "DataOnNodes");
      int i_field = FieldNameToId(i_file, i_base, i_zone, i_sol, "MetisIndex");
      if (cgp_field_general_read_data(i_file, i_base, i_zone, i_sol, i_field,
          range_min, range_max, kIntType,
          1, mem_dimensions, mem_range_min, mem_range_max, metis_id.data()))
        cgp_error_exit();
      for (int i_node = head; i_node < tail; ++i_node) {
        auto m_node = metis_id[i_node];
        m_to_node_info_[m_node] = NodeInfo(i_zone, i_node);
      }
    }
  }
  std::pair<
    std::map<Int, std::vector<Int>>,
    std::vector<std::vector<Scalar>>
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
    std::vector<std::vector<Scalar>> send_bufs;
    for (auto& [i_part, nodes] : send_nodes) {
      auto& coords = send_bufs.emplace_back();
      for (auto m_node : nodes) {
        auto& info = m_to_node_info_.at(m_node);
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
      m_to_node_info_[m_node] = cgns::NodeInfo<Int>(i_zone, i_node);
    }
    std::vector<std::vector<Scalar>> recv_coords;
    for (auto& [i_part, nodes] : recv_nodes) {
      assert(std::is_sorted(nodes.begin(), nodes.end()));
      int n_reals = 3 * nodes.size();
      auto& coords = recv_coords.emplace_back(std::vector<Scalar>(n_reals));
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
      const std::vector<std::vector<Scalar>>& recv_coords) {
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
  auto BuildTetraUptr(int i_zone, const Int *i_node_list) const {
    return std::make_unique<integrator::Tetra<Scalar, 24>>(
        GetCoord(i_zone, i_node_list[0]), GetCoord(i_zone, i_node_list[1]),
        GetCoord(i_zone, i_node_list[2]), GetCoord(i_zone, i_node_list[3]));
  }
  auto BuildHexaUptr(int i_zone, const Int *i_node_list) const {
    return std::make_unique<integrator::Hexa<Scalar, 4, 4, 4>>(
        GetCoord(i_zone, i_node_list[0]), GetCoord(i_zone, i_node_list[1]),
        GetCoord(i_zone, i_node_list[2]), GetCoord(i_zone, i_node_list[3]),
        GetCoord(i_zone, i_node_list[4]), GetCoord(i_zone, i_node_list[5]),
        GetCoord(i_zone, i_node_list[6]), GetCoord(i_zone, i_node_list[7]));
  }
  auto BuildGaussForCell(int npe, int i_zone, const Int *i_node_list) const {
    std::unique_ptr<integrator::Cell<Scalar>> gauss_uptr;
    switch (npe) {
      case 4:
        gauss_uptr = BuildTetraUptr(i_zone, i_node_list); break;
      case 8:
        gauss_uptr = BuildHexaUptr(i_zone, i_node_list); break;
      default:
        assert(false);
        break;
    }
    return gauss_uptr;
  }
  auto BuildTriUptr(int i_zone, const Int *i_node_list) const {
    auto quad_uptr = std::make_unique<integrator::Tri<Scalar, kDim, 16>>(
        GetCoord(i_zone, i_node_list[0]), GetCoord(i_zone, i_node_list[1]),
        GetCoord(i_zone, i_node_list[2]));
    quad_uptr->BuildNormalFrames();
    return quad_uptr;
  }
  auto BuildQuadUptr(int i_zone, const Int *i_node_list) const {
    auto quad_uptr = std::make_unique<integrator::Quad<Scalar, kDim, 4, 4>>(
        GetCoord(i_zone, i_node_list[0]), GetCoord(i_zone, i_node_list[1]),
        GetCoord(i_zone, i_node_list[2]), GetCoord(i_zone, i_node_list[3]));
    quad_uptr->BuildNormalFrames();
    return quad_uptr;
  }
  auto BuildGaussForFace(int npe, int i_zone, const Int *i_node_list) const {
    std::unique_ptr<integrator::Face<Scalar, kDim>> gauss_uptr;
    switch (npe) {
      case 3:
        gauss_uptr = BuildTriUptr(i_zone, i_node_list); break;
      case 4:
        gauss_uptr = BuildQuadUptr(i_zone, i_node_list); break;
      default:
        assert(false);
        break;
    }
    return gauss_uptr;
  }
  static void SortNodesOnFace(int npe, const Int *cell, Int *face) {
    switch (npe) {
      case 4:
        integrator::Tetra<Scalar, 24>::SortNodesOnFace(cell, face);
        break;
      case 8:
        integrator::Hexa<Scalar, 4, 4, 4>::SortNodesOnFace(cell, face);
        break;
      default:
        assert(false);
        break;
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
      cgsize_t range_min[] = { head };
      cgsize_t range_max[] = { tail - 1 };
      cgsize_t mem_dimensions[] = { tail - head };
      cgsize_t mem_range_min[] = { 1 };
      cgsize_t mem_range_max[] = { mem_dimensions[0] };
      ShiftedVector<Int> metis_ids(mem_dimensions[0], head);
      int i_sol = SolnNameToId(i_file, i_base, i_zone, "DataOnCells");
      int i_field = FieldNameToId(i_file, i_base, i_zone, i_sol, "MetisIndex");
      if (cgp_field_general_read_data(i_file, i_base, i_zone, i_sol, i_field,
          range_min, range_max, kIntType,
          1, mem_dimensions, mem_range_min, mem_range_max, metis_ids.data()))
        cgp_error_exit();
      char name[33];
      ElementType_t type;
      cgsize_t u, v;
      int x, y;
      auto& conn = cell_conn[i_zone][i_sect];
      auto& index = conn.index;
      auto& nodes = conn.nodes;
      index = ShiftedVector<Int>(mem_dimensions[0] + 1, head);
      if (cg_section_read(i_file, i_base, i_zone, i_sect, name, &type,
          &u, &v, &x, &y))
        cgp_error_exit();
      int npe; cg_npe(type, &npe);
      for (int i_cell = head; i_cell < tail; ++i_cell) {
        auto m_cell = metis_ids[i_cell];
        m_to_cell_info_[m_cell] = CellInfo(i_zone, i_sect, i_cell, npe);
      }
      nodes.resize(npe * mem_dimensions[0]);
      for (int i = 0; i < index.size(); ++i) {
        index.at(head + i) = npe * i;
      }
      if (cgp_elements_read_data(i_file, i_base, i_zone, i_sect,
          range_min[0], range_max[0], nodes.data()))
        cgp_error_exit();
      local_cells_[i_zone][i_sect] = CellGroup(head, tail - head, npe);
      for (int i_cell = head; i_cell < tail; ++i_cell) {
        auto* i_node_list = &nodes[(i_cell - head) * npe];
        auto gauss_uptr = BuildGaussForCell(npe, i_zone, i_node_list);
        auto cell = Cell(std::move(gauss_uptr), metis_ids[i_cell]);
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
        send_npes, recv_npes;  // [i_part][m_cell] -> npe
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
    auto& send_npes = ghost_adj.send_npes;
    auto& recv_npes = ghost_adj.recv_npes;
    auto& m_cell_pairs = ghost_adj.m_cell_pairs;
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int p, i, j, npe_i, npe_j;
      std::sscanf(line, "%d %d %d %d %d", &p, &i, &j, &npe_i, &npe_j);
      send_npes[p][i] = npe_i;
      recv_npes[p][j] = npe_j;
      m_cell_pairs.emplace_back(i, j);
    }
    return ghost_adj;
  }
  auto ShareGhostCells(const GhostAdj& ghost_adj,
      const ZoneSectToConn& cell_conn) {
    auto& send_npes = ghost_adj.send_npes;
    auto& recv_npes = ghost_adj.recv_npes;
    // send cell.i_zone and cell.node_id_list
    std::vector<std::vector<Int>> send_cells;
    std::vector<MPI_Request> requests;
    for (auto& [i_part, npes] : send_npes) {
      auto& send_buf = send_cells.emplace_back();
      for (auto& [m_cell, npe] : npes) {
        auto& info = m_to_cell_info_[m_cell];
        assert(npe == info.npe);
        Int i_zone = info.i_zone, i_sect = info.i_sect, i_cell = info.i_cell;
        auto& conn = cell_conn.at(i_zone).at(i_sect);
        auto& index = conn.index;
        auto& nodes = conn.nodes;
        auto* i_node_list = &(nodes[index[i_cell]]);
        send_buf.emplace_back(i_zone);
        for (int i = 0; i < npe; ++i) {
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
    for (auto& [i_part, npes] : recv_npes) {
      auto& recv_buf = recv_cells.emplace_back();
      int n_ints = 0;
      for (auto& [m_cell, npe] : npes) {
        ++n_ints;
        n_ints += npe;
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
  struct GhostCellInfo {
    int source, head, npe;
  };
  std::unordered_map<Int, GhostCellInfo> BuildGhostCells(
      const GhostAdj& ghost_adj,
      const std::vector<std::vector<Int>>& recv_cells) {
    auto& recv_npes = ghost_adj.recv_npes;
    // build ghost cells
    std::unordered_map<Int, GhostCellInfo> m_to_recv_cells;
    int i_source = 0;
    for (auto& [i_part, npes] : recv_npes) {
      auto& recv_buf = recv_cells.at(i_source);
      int index = 0;
      for (auto& [m_cell, npe] : npes) {
        m_to_recv_cells[m_cell].source = i_source;
        m_to_recv_cells[m_cell].head = index + 1;
        m_to_recv_cells[m_cell].npe = npe;
        int i_zone = recv_buf[index++];
        auto* i_node_list = &recv_buf[index];
        auto gauss_uptr = BuildGaussForCell(npe, i_zone, i_node_list);
        auto cell = Cell(std::move(gauss_uptr), m_cell);
        ghost_cells_[m_cell] = std::move(cell);
        index += npe;
      }
      ++i_source;
    }
    return m_to_recv_cells;
  }
  void FillCellPtrs(const GhostAdj& ghost_adj) {
    // fill `send_cell_ptrs_`
    for (auto& [i_part, npes] : ghost_adj.send_npes) {
      auto& curr_part = send_cell_ptrs_[i_part];
      assert(curr_part.empty());
      for (auto& [m_cell, npe] : npes) {
        auto& info = m_to_cell_info_[m_cell];
        auto& cell = local_cells_.at(info.i_zone).at(info.i_sect)[info.i_cell];
        cell.inner_ = false;
        curr_part.emplace_back(&cell);
      }
      send_coeffs_.emplace_back(npes.size() * kFields);
    }
    // fill `recv_cell_ptrs_`
    for (auto& [i_part, npes] : ghost_adj.recv_npes) {
      auto& curr_part = recv_cell_ptrs_[i_part];
      assert(curr_part.empty());
      for (auto& [m_cell, npe] : npes) {
        auto& cell = ghost_cells_.at(m_cell);
        curr_part.emplace_back(&cell);
      }
      recv_coeffs_.emplace_back(npes.size() * kFields);
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
      for (int i = 0; i < holder_info.npe; ++i)
        ++i_node_cnt[holder_nodes[holder_head + i]];
      for (int i = 0; i < sharer_info.npe; ++i)
        ++i_node_cnt[sharer_nodes[sharer_head + i]];
      auto common_nodes = std::vector<Int>();
      common_nodes.reserve(4);
      for (auto [i_node, cnt] : i_node_cnt)
        if (cnt == 2)
          common_nodes.emplace_back(i_node);
      int face_npe = common_nodes.size();
      // let the normal vector point from holder to sharer
      // see http://cgns.github.io/CGNS_docs_current/sids/conv.figs/hexa_8.png
      auto& zone = local_cells_[i_zone];
      auto& holder = zone[holder_info.i_sect][holder_info.i_cell];
      auto& sharer = zone[sharer_info.i_sect][sharer_info.i_cell];
      holder.adj_cells_.emplace_back(&sharer);
      sharer.adj_cells_.emplace_back(&holder);
      auto* i_node_list = common_nodes.data();
      SortNodesOnFace(holder_info.npe, &holder_nodes[holder_head], i_node_list);
      auto gauss_uptr = BuildGaussForFace(face_npe, i_zone, i_node_list);
      auto face_uptr = std::make_unique<Face>(
          std::move(gauss_uptr), &holder, &sharer, local_faces_.size());
      holder.adj_faces_.emplace_back(face_uptr.get());
      sharer.adj_faces_.emplace_back(face_uptr.get());
      local_faces_.emplace_back(std::move(face_uptr));
    }
  }
  void BuildGhostFaces(const GhostAdj& ghost_adj,
      const ZoneSectToConn& cell_conn,
      const std::vector<std::vector<Int>>& recv_cells,
      const std::unordered_map<Int, GhostCellInfo>& m_to_recv_cells) {
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
      auto& sharer_nodes = recv_cells[sharer_info.source];
      auto holder_head = holder_conn.index[holder_info.i_cell];
      auto sharer_head = sharer_info.head;
      for (int i = 0; i < holder_info.npe; ++i)
        ++i_node_cnt[holder_nodes[holder_head + i]];
      for (int i = 0; i < sharer_info.npe; ++i)
        ++i_node_cnt[sharer_nodes[sharer_head + i]];
      auto common_nodes = std::vector<Int>();
      common_nodes.reserve(4);
      for (auto [i_node, cnt] : i_node_cnt) {
        if (cnt == 2)
          common_nodes.emplace_back(i_node);
      }
      int face_npe = common_nodes.size();
      // let the normal vector point from holder to sharer
      // see http://cgns.github.io/CGNS_docs_current/sids/conv.figs/hexa_8.png
      auto& zone = local_cells_[i_zone];
      auto& holder = zone[holder_info.i_sect][holder_info.i_cell];
      auto& sharer = ghost_cells_.at(m_sharer);
      holder.adj_cells_.emplace_back(&sharer);
      auto* i_node_list = common_nodes.data();
      SortNodesOnFace(holder_info.npe, &holder_nodes[holder_head], i_node_list);
      auto gauss_uptr = BuildGaussForFace(face_npe, i_zone, i_node_list);
      auto face_uptr = std::make_unique<Face>(
          std::move(gauss_uptr), &holder, &sharer,
          local_faces_.size() + ghost_faces_.size());
      holder.adj_faces_.emplace_back(face_uptr.get());
      ghost_faces_.emplace_back(std::move(face_uptr));
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
  void ScatterSolutions() {
    int n_zones = local_nodes_.size();
    for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
      int n_sects = local_cells_[i_zone].size();
      for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
        local_cells_[i_zone][i_sect].ScatterFields();
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
      for (int i_field = 1; i_field <= kFields; ++i_field) {
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
  void ReadSolutions(const std::string &soln_name) {
    int n_zones = local_nodes_.size();
    int i_file;
    if (cgp_open(cgns_file_.c_str(), CG_MODE_READ, &i_file))
      cgp_error_exit();
    for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
      auto& zone = local_cells_.at(i_zone);
      int n_solns;
      if (cg_nsols(i_file, i_base, i_zone, &n_solns))
        cgp_error_exit();
      int i_soln;
      char name[33];
      GridLocation_t lc;
      for (i_soln = 1; i_soln <= n_solns; ++i_soln) {
        if (cg_sol_info(i_file, i_base, i_zone, i_soln, name, &lc))
          cgp_error_exit();
        if (name == soln_name)
          break;
      }
      int n_fields;
      if (cg_nfields(i_file, i_base, i_zone, i_soln, &n_fields)) {
        cgp_error_exit();
      }
      assert(n_fields == kFields);
      for (int i_field = 1; i_field <= n_fields; ++i_field) {
        int n_sects = zone.size();
        for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
          auto& section = zone.at(i_sect);
          char field_name[33];
          DataType_t data_type;
          if (cg_field_info(i_file, i_base, i_zone, i_soln, i_field,
              &data_type, field_name))
            cgp_error_exit();
          cgsize_t first[] = { section.head() };
          cgsize_t last[] = { section.tail() - 1 };
          if (cgp_field_read_data(i_file, i_base, i_zone, i_soln, i_field,
              first, last, section.GetField(i_field).data()))
            cgp_error_exit();
        }
      }
    }
    if (cgp_close(i_file))
      cgp_error_exit();
  }
  void WriteSolutionsOnCellCenters(const std::string &soln_name = "0") const {
    // prepare data to be written
    Int n_cells = 0;
    auto coords = std::vector<Mat3x1>();
    auto fields = std::vector<typename Cell::Value>();
    for (auto& [i_zone, zone] : local_cells_) {
      for (auto& [i_sect, sect] : zone) {
        int i_cell = sect.head();
        int i_cell_tail = sect.tail();
        int type = Cell::GetVtkType(sect.npe());
        while (i_cell < i_cell_tail) {
          sect[i_cell].WriteSolutions(type, &coords, &fields);
          ++i_cell; ++n_cells;
        }
      }
    }
    if (rank_ == 0) {
      char temp[1024];
      std::snprintf(temp, sizeof(temp), "%s/%s.pvtu",
          directory_.c_str(), soln_name.c_str());
      auto pvtu = std::ofstream(temp, std::ios::out);
      pvtu << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" "
          << "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
      pvtu << "  <PUnstructuredGrid GhostLevel=\"1\">\n";
      pvtu << "    <PPointData>\n";
      for (int k = 0; k < kFunc; ++k) {
        pvtu << "      <PDataArray type=\"Float64\" Name=\""
            << field_names_[k] << "\"/>\n";
      }
      pvtu << "    </PPointData>\n";
      pvtu << "    <PPoints>\n";
      pvtu << "      <PDataArray type=\"Float64\" Name=\"Points\" "
          << "NumberOfComponents=\"3\"/>\n";
      pvtu << "    </PPoints>\n";
      int n_parts;
      MPI_Comm_size(MPI_COMM_WORLD, &n_parts);
      for (int i_part = 0; i_part < n_parts; ++i_part) {
        pvtu << "    <Piece Source=\"./" << soln_name << '/'
            << i_part << ".vtu\"/>\n";
      }
      pvtu << "  </PUnstructuredGrid>\n";
      pvtu << "</VTKFile>\n";
    }
    // write to an ofstream
    bool binary = false;
    auto vtu = GetFileStream(soln_name, binary, "vtu");
    vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\""
        << " byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    vtu << "  <UnstructuredGrid>\n";
    vtu << "    <Piece NumberOfPoints=\"" << coords.size()
        << "\" NumberOfCells=\"" << n_cells << "\">\n";
    vtu << "      <PointData>\n";
    int K = fields[0].size();
    for (int k = 0; k < K; ++k) {
      vtu << "        <DataArray type=\"Float64\" Name=\""
          << field_names_[k] << "\" format=\"ascii\">\n";
      for (auto& f : fields) {
        vtu << f[k] << ' ';
      }
      vtu << "\n        </DataArray>\n";
    }
    vtu << "      </PointData>\n";
    vtu << "      <DataOnCells>\n";
    vtu << "      </DataOnCells>\n";
    vtu << "      <Points>\n";
    vtu << "        <DataArray type=\"Float64\" Name=\"Points\" "
        << "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (auto& xyz : coords) {
      for (auto v : xyz) {
        vtu << v << ' ';
      }
    }
    vtu << "\n        </DataArray>\n";
    vtu << "      </Points>\n";
    vtu << "      <Cells>\n";
    vtu << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
        << "format=\"ascii\">\n";
    for (int i_node = 0; i_node < coords.size(); ++i_node) {
      vtu << i_node << ' ';
    }
    vtu << "\n        </DataArray>\n";
    vtu << "        <DataArray type=\"Int32\" Name=\"offsets\" "
        << "format=\"ascii\">\n";
    int offset = 0;
    for (auto& [i_zone, zone] : local_cells_) {
      for (auto& [i_sect, sect] : zone) {
        int i_cell = sect.head();
        int i_cell_tail = sect.tail();
        int n_nodes = Cell::CountNodes(Cell::GetVtkType(sect.npe()));
        while (i_cell < i_cell_tail) {
          offset += n_nodes;
          vtu << offset << ' ';
          ++i_cell;
        }
      }
    }
    vtu << "\n        </DataArray>\n";
    vtu << "        <DataArray type=\"UInt8\" Name=\"types\" "
        << "format=\"ascii\">\n";
    for (auto& [i_zone, zone] : local_cells_) {
      for (auto& [i_sect, sect] : zone) {
        int i_cell = sect.head();
        int i_cell_tail = sect.tail();
        auto type = Cell::GetVtkType(sect.npe());
        while (i_cell < i_cell_tail) {
          vtu << type << ' ';
          ++i_cell;
        }
      }
    }
    vtu << "\n        </DataArray>\n";
    vtu << "      </Cells>\n";
    vtu << "    </Piece>\n";
    vtu << "  </UnstructuredGrid>\n";
    vtu << "</VTKFile>\n";
  }
  template <class V>
  static void WriteInBigEndian(const V& v, std::ofstream &ostrm) {
    char a[sizeof(v)];
    auto *pv = reinterpret_cast<const char*>(&v);
    for (int i = 0; i < sizeof(v); ++i) {
      a[i] = pv[sizeof(v) - 1 - i];
    }
    ostrm.write(a, sizeof(v));
  }
  void ShareGhostCellCoeffs() {
    int i_req = 0;
    // send cell.projection_.coeff_
    int i_buf = 0;
    for (auto& [i_part, cell_ptrs] : send_cell_ptrs_) {
      auto& send_buf = send_coeffs_[i_buf++];
      int i_real = 0;
      for (auto* cell_ptr : cell_ptrs) {
        const auto& coeff = cell_ptr->projection_.coeff();
        static_assert(kFields == Cell::K * Cell::N);
        for (int c = 0; c < Cell::N; ++c) {
          for (int r = 0; r < Cell::K; ++r) {
            send_buf[i_real++] = coeff(r, c);
          }
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
    auto new_projections = std::vector<typename Cell::Projection>();
    new_projections.reserve(inner_cells_.size());
    for (const auto* cell_ptr : inner_cells_) {
      new_projections.emplace_back(limiter(*cell_ptr));
    }
    int i = 0;
    for (auto* cell_ptr : inner_cells_) {
      cell_ptr->projection_.UpdateCoeffs(new_projections[i++].coeff());
    }
    // run the limiter on inter cells that need ghost cells
    new_projections.clear();
    new_projections.reserve(inter_cells_.size());
    UpdateGhostCellCoeffs();
    for (const auto* cell_ptr : inter_cells_) {
      new_projections.emplace_back(limiter(*cell_ptr));
    }
    i = 0;
    for (auto* cell_ptr : inter_cells_) {
      cell_ptr->projection_.UpdateCoeffs(new_projections[i++].coeff());
    }
  }

  // Accessors:
  template<class Visitor>
  void ForEachConstLocalCell(Visitor&& visit) const {
    for (const auto& [i_zone, zone] : local_cells_) {
      for (const auto& [i_sect, sect] : zone) {
        for (const auto& cell : sect) {
          visit(cell);
        }
      }
    }
  }
  template<class Visitor>
  void ForEachConstLocalFace(Visitor&& visit) const {
    for (auto& face_uptr : local_faces_) {
      visit(*face_uptr);
    }
  }
  template<class Visitor>
  void ForEachConstGhostFace(Visitor&& visit) const {
    for (auto& face_uptr : ghost_faces_) {
      visit(*face_uptr);
    }
  }
  template<class Visitor>
  void ForEachConstBoundaryFace(Visitor&& visit) const {
    for (auto& [i_zone, zone] : bound_faces_) {
      for (auto& [i_sect, sect] : zone) {
        for (auto& face_uptr : sect) {
          visit(*face_uptr);
        }
      }
    }
  }
  template<class Visitor>
  void ForEachConstBoundaryFace(Visitor&& visit,
      const std::string &name) const {
    const auto& faces = *name_to_faces_.at(name);
    for (const auto& face_uptr : faces) {
      visit(*face_uptr);
    }
  }
  // Mutators:
  template<class Visitor>
  void ForEachLocalCell(Visitor&& visit) {
    for (auto& [i_zone, zone] : local_cells_) {
      for (auto& [i_sect, sect] : zone) {
        for (auto& cell : sect) {
          visit(&cell);
        }
      }
    }
  }
  template<class Visitor>
  void ForEachLocalFace(Visitor&& visit) {
    for (auto& face_uptr : local_faces_) {
      visit(face_uptr.get());
    }
  }
  template<class Visitor>
  void ForEachGhostFace(Visitor&& visit) {
    for (auto& face_uptr : ghost_faces_) {
      visit(face_uptr.get());
    }
  }
  template<class Visitor>
  void ForEachBoundaryFace(Visitor&& visit) {
    for (auto& [i_zone, zone] : bound_faces_) {
      for (auto& [i_sect, sect] : zone) {
        for (auto& face_uptr : sect) {
          visit(face_uptr.get());
        }
      }
    }
  }

 private:
  std::map<Int, NodeGroup>
      local_nodes_;  // [i_zone] -> a NodeGroup obj
  std::unordered_map<Int, std::unordered_map<Int, Mat3x1>>
      ghost_nodes_;  // [i_zone][i_node] -> a Mat3x1 obj
  std::unordered_map<Int, NodeInfo>
      m_to_node_info_;  // [m_node] -> a NodeInfo obj
  std::unordered_map<Int, CellInfo>
      m_to_cell_info_;  // [m_cell] -> a CellInfo obj
  std::map<Int, std::map<Int, CellGroup>>
      local_cells_;  // [i_zone][i_sect][i_cell] -> a Cell obj
  std::vector<Cell*>
      inner_cells_, inter_cells_;  // [i_cell] -> Cell*
  std::map<Int, std::vector<Cell*>>
      send_cell_ptrs_, recv_cell_ptrs_;  // [i_part] -> vector<Cell*>
  std::vector<std::vector<Scalar>>
      send_coeffs_, recv_coeffs_;
  std::unordered_map<Int, Cell>
      ghost_cells_;  // [m_cell] -> a Cell obj
  std::vector<std::pair<Int, Int>>
      local_adjs_;  // [i_pair] -> { m_holder, m_sharer }
  std::vector<std::unique_ptr<Face>>
      local_faces_, ghost_faces_;  // [i_face] -> a uptr of Face
  std::map<Int, std::map<Int, ShiftedVector<std::unique_ptr<Face>>>>
      bound_faces_;  // [i_zone][i_sect][i_face] -> a uptr of Face
  std::unordered_map<std::string, ShiftedVector<std::unique_ptr<Face>> *>
      name_to_faces_;
  std::vector<MPI_Request> requests_;
  std::array<std::string, kFunc> field_names_;
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
    Int face_id = local_faces_.size() + ghost_faces_.size();
    // build boundary faces
    std::unordered_map<std::string, std::pair<int, int>>
        name_to_z_s;  // name -> { i_zone, i_sect }
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
      int x, y;
      auto& conn = face_conn[i_zone][i_sect];
      auto& index = conn.index;
      auto& nodes = conn.nodes;
      if (cg_section_read(i_file, i_base, i_zone, i_sect, name, &type,
          &u, &v, &x, &y))
        cgp_error_exit();
      name_to_z_s[name] = { i_zone, i_sect };
      int npe; cg_npe(type, &npe);
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
        Cell *holder_ptr;
        for (auto [m_cell, cnt] : cell_cnt) {
          assert(cnt <= npe);
          if (cnt == npe) {  // this cell holds this face
            auto& info = m_to_cell_info_[m_cell];
            Int z = info.i_zone, s = info.i_sect, c = info.i_cell;
            holder_ptr = &(local_cells_.at(z).at(s).at(c));
            auto& holder_conn = cell_conn.at(z).at(s);
            auto& holder_nodes = holder_conn.nodes;
            auto holder_head = holder_conn.index[c];
            SortNodesOnFace(info.npe, &holder_nodes[holder_head], i_node_list);
            break;
          }
        }
        auto gauss_uptr = BuildGaussForFace(npe, i_zone, i_node_list);
        auto face_uptr = std::make_unique<Face>(
            std::move(gauss_uptr), holder_ptr, nullptr, face_id++);
        // holder_ptr->adj_faces_.emplace_back(face_uptr.get());
        faces.emplace_back(std::move(face_uptr));
      }
    }
    // build name to ShiftedVector of faces
    for (auto& [name, z_s] : name_to_z_s) {
      auto [i_zone, i_sect] = z_s;
      auto& faces = bound_faces_.at(i_zone).at(i_sect);
      name_to_faces_[name] = &faces;
    }
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
  std::ofstream GetFileStream(const std::string &soln_name, bool binary,
      const std::string &suffix) const {
    char temp[1024];
    if (rank_ == 0) {
      std::snprintf(temp, sizeof(temp), "mkdir -p %s/%s",
          directory_.c_str(), soln_name.c_str());
      std::system(temp);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::snprintf(temp, sizeof(temp), "%s/%s/%d.%s",
        directory_.c_str(), soln_name.c_str(), rank_, suffix.c_str());
    return std::ofstream(temp,
        std::ios::out | (binary ? (std::ios::binary) : std::ios::out));
  }
};
template <typename Int, int kOrder, class R>
MPI_Datatype const Part<Int, kOrder, R>::kMpiIntType
    = sizeof(Int) == 8 ? MPI_LONG : MPI_INT;
template <typename Int, int kOrder, class R>
MPI_Datatype const Part<Int, kOrder, R>::kMpiRealType
    = sizeof(Scalar) == 8 ? MPI_DOUBLE : MPI_FLOAT;

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_PART_HPP_
