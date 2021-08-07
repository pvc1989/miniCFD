// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan
/**
 * This file defines wrappers of APIs and types in CGNS/MLL.
 */
#ifndef MINI_MESH_CGNS_TYPES_HPP_
#define MINI_MESH_CGNS_TYPES_HPP_

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

/**
 * Return true if the cell type is supported and consistent with the given dim.
 */
inline bool CheckTypeDim(CGNS_ENUMT(ElementType_t) type, int cell_dim) {
  if (cell_dim == 2) {
    if (type == CGNS_ENUMV(TRI_3) || type == CGNS_ENUMV(QUAD_4) ||
        type == CGNS_ENUMV(MIXED))
      return true;
  } else if (cell_dim == 3) {
    if (type == CGNS_ENUMV(TETRA_4) || type == CGNS_ENUMV(HEXA_8) ||
        type == CGNS_ENUMV(MIXED))
      return true;
  }
  return false;
}

/**
 * Wrapper of the `GridCoordinates_t` type.
 */
template <class Real>
struct Coordinates {
 public:  // Constructors:
  explicit Coordinates(int fid, int bid, int zid, int size)
      : file_id_{fid}, base_id_{bid}, zone_id_{zid},
        x_(size), y_(size), z_(size), name_("GridCoordinates") {
  }

 public:  // Copy Control:
  Coordinates(Coordinates const &) = default;
  Coordinates& operator=(const Coordinates&) = default;
  Coordinates(Coordinates&&) noexcept = default;
  Coordinates& operator=(Coordinates&&) noexcept = default;
  ~Coordinates() noexcept = default;

 public:  // Accessors:
  int CountNodes() const {
    return x_.size();
  }
  std::string const& name() const {
    return name_;
  }
  std::vector<Real> const& x() const {
    return x_;
  }
  std::vector<Real> const& y() const {
    return y_;
  }
  std::vector<Real> const& z() const {
    return z_;
  }

 public:  // Mutators:
  /**
   * Read coordinates from a given `(file, base, zone)` tuple.
   */
  void Read() {
    // All id's are 1-based when passing to CGNS/MLL.
    cgsize_t first = 1;
    cgsize_t last = CountNodes();
    auto data_type = std::is_same_v<Real, double> ?
        CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
    cg_coord_read(file_id_, base_id_, zone_id_, "CoordinateX",
                  data_type, &first, &last, x_.data());
    cg_coord_read(file_id_, base_id_, zone_id_, "CoordinateY",
                  data_type, &first, &last, y_.data());
    cg_coord_read(file_id_, base_id_, zone_id_, "CoordinateZ",
                  data_type, &first, &last, z_.data());
  }
  /**
   * Write coordinates to a given `(file, base, zone)` tuple.
   */
  void Write() {
    int coord_id;
    auto data_type = std::is_same_v<Real, double> ?
        CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
    cg_coord_write(file_id_, base_id_, zone_id_,
                   data_type, "CoordinateX", x_.data(), &coord_id);
    cg_coord_write(file_id_, base_id_, zone_id_,
                   data_type, "CoordinateY", y_.data(), &coord_id);
    cg_coord_write(file_id_, base_id_, zone_id_,
                   data_type, "CoordinateZ", z_.data(), &coord_id);
  }

 private:  // Data Members:
  std::string name_;
  std::vector<Real> x_, y_, z_;
  int file_id_, base_id_, zone_id_;
};

/**
 * Wrapper of the `Elements_t` type.
 */
template <class Real>
struct Section {
 public:  // Constructors:
  Section(int fid, int bid, int zid, int sid,
          char const* name, cgsize_t first, cgsize_t size,
          int n_boundary_cells, CGNS_ENUMT(ElementType_t) type)
      : file_id_{fid}, base_id_{bid}, zone_id_{zid},
        section_id_{sid}, name_{name}, first_{first}, size_{size},
        n_boundary_cells_{n_boundary_cells}, type_{type},
        node_id_list_(size * CountNodesByType(type)) {
  }

 public:  // Copy Control:
  Section(const Section&) = default;
  Section& operator=(const Section&) = default;
  Section(Section&&) noexcept = default;
  Section& operator=(Section&&) noexcept = default;
  ~Section() noexcept = default;

 public:  // Accessors:
  std::string const& name() const { return name_; }
  int id() const { return section_id_; }
  cgsize_t CellIdMin() const { return first_; }
  cgsize_t CellIdMax() const { return first_ + size_ - 1; }
  cgsize_t CountCells() const { return size_; }
  CGNS_ENUMT(ElementType_t) type() const {
    return type_;
  }
  cgsize_t* GetNodeIdList() {
    return node_id_list_.data();
  }
  const cgsize_t* GetNodeIdList() const {
    return node_id_list_.data();
  }
  cgsize_t* GetNodeIdListByNilBasedRow(cgsize_t row) {
    return node_id_list_.data() + CountNodesByType(type_) * row;
  }
  const cgsize_t* GetNodeIdListByNilBasedRow(cgsize_t row) const {
    return node_id_list_.data() + CountNodesByType(type_) * row;
  }
  cgsize_t* GetNodeIdListByOneBasedCellId(cgsize_t cell_id) {
    return GetNodeIdListByNilBasedRow(cell_id - first_);
  }
  const cgsize_t* GetNodeIdListByOneBasedCellId(cgsize_t cell_id) const {
    return GetNodeIdListByNilBasedRow(cell_id - first_);
  }
  static int CountNodesByType(CGNS_ENUMT(ElementType_t) type) {
    int npe;
    cg_npe(type, &npe);
    return npe;
  }

 public:  // Mutators:
  /**
   * Read node_id_list from a given `(file, base, zone)` tuple.
   */
  void Read() {
    cg_elements_read(file_id_, base_id_, zone_id_, section_id_,
                     GetNodeIdList(), NULL/* int* parent_data */);
  }
  /**
   * Write node_id_list into a given `(file, base, zone)` tuple.
   */
  void Write() {
    int section_id;
    cg_section_write(file_id_, base_id_, zone_id_, name_.c_str(), type_,
        CellIdMin(), CellIdMax(), 0, GetNodeIdList(),
        &section_id);
    assert(section_id == section_id_);
  }

 private:  // Data Members:
  std::vector<cgsize_t> node_id_list_;
  std::vector<cgsize_t> start_offset_;
  std::string name_;
  int file_id_, base_id_, zone_id_, section_id_, n_boundary_cells_;
  cgsize_t first_, size_;
  CGNS_ENUMT(ElementType_t) type_;
};

template <class T>
using Field = std::vector<T>;

template <class Real>
struct Solution {
  Solution(int fid, int bid, int zid, int sid, char const* name,
           CGNS_ENUMT(GridLocation_t) location)
      : file_id_(fid), base_id_(bid), zone_id_(zid),
        sol_id_(sid), name_(name), location_(location) {
  }
  int id() const {
    return sol_id_;
  }
  std::string const& name() const {
    return name_;
  }
  std::map<std::string, Field<Real>> const& fields() const {
    return fields_;
  }
  std::map<std::string, Field<Real>>& fields() {
    return fields_;
  }

  void Write() {
    int sol_id;
    cg_sol_write(file_id_, base_id_, zone_id_,
                 name_.c_str(), location_, &sol_id);
    assert(sol_id == sol_id_);
    for (auto& [field_name, field] : fields_) {
      int field_id;
      cg_field_write(file_id_, base_id_, zone_id_, sol_id,
                     CGNS_ENUMV(RealDouble),
                     field_name.c_str(), field.data(), &field_id);
    }
  }


 private:
  std::map<std::string, Field<Real>> fields_;
  std::string name_;
  CGNS_ENUMT(GridLocation_t) location_;
  int file_id_, base_id_, zone_id_, sol_id_;
};

template <class Real>
class Zone {
 public:
  using CoordinatesType = Coordinates<Real>;
  using SectionType = Section<Real>;
  using SolutionType = Solution<Real>;

  Zone(int fid, int bid, int zid, char const* name,
       cgsize_t n_cells, cgsize_t n_nodes)
      : file_id_(fid), base_id_(bid), zone_id_(zid),
        name_(name), cell_size_(n_cells),
        coordinates_(fid, bid, zid, n_nodes) {
  }
  int id() const {
    return zone_id_;
  }
  const std::string& name() const {
    return name_;
  }
  int CountNodes() const {
    return coordinates_.CountNodes();
  }
  int CountCells() const {
    return cell_size_;
  }
  int CountCellsByType(CGNS_ENUMT(ElementType_t) type) const {
    int cell_size{0};
    for (auto& section : sections_) {
      if (section.type == type) {
        cell_size += section.CountCells();
      }
    }
    return cell_size;
  }
  int CountSections() const {
    return sections_.size();
  }
  int CountSolutions() const {
    return solutions_.size();
  }
  CoordinatesType& GetCoordinates() {
    return coordinates_;
  }
  SectionType& GetSection(int id) {
    return sections_.at(id-1);
  }
  SolutionType& GetSolution(int id) {
    return solutions_.at(id-1);
  }
  const CoordinatesType& GetCoordinates() const {
    return coordinates_;
  }
  const SectionType& GetSection(int id) const {
    return sections_.at(id-1);
  }
  const SolutionType& GetSolution(int id) const {
    return solutions_.at(id-1);
  }

 public:  // Mutators:
  void ReadCoordinates() {
    coordinates_.Read();
  }
  void ReadAllSections() {
    int n_sections;
    cg_nsections(file_id_, base_id_, zone_id_, &n_sections);
    sections_.reserve(n_sections);
    for (int section_id = 1; section_id <= n_sections; ++section_id) {
      char section_name[33];
      CGNS_ENUMT(ElementType_t) cell_type;
      cgsize_t first, last;
      int n_boundary_cells, parent_flag;
      cg_section_read(file_id_, base_id_, zone_id_, section_id,
                      section_name, &cell_type, &first, &last,
                      &n_boundary_cells, &parent_flag);
      auto& section = sections_.emplace_back(
          file_id_, base_id_, zone_id_, section_id,
          section_name, first, /* size = */last - first + 1,
          n_boundary_cells, cell_type);
      section.Read();
    }
  }
  void ReadSectionsWithDim(int cell_dim) {
    int n_sections;
    cg_nsections(file_id_, base_id_, zone_id_, &n_sections);
    cgsize_t range_min{1};
    int new_section_id{1};
    for (int section_id = 1; section_id <= n_sections; ++section_id) {
      char section_name[33]; CGNS_ENUMT(ElementType_t) cell_type;
      cgsize_t first, last; int n_boundary_cells, parent_flag;
      cg_section_read(file_id_, base_id_, zone_id_, section_id, section_name,
                      &cell_type, &first, &last, &n_boundary_cells,
                      &parent_flag);
      if (!CheckTypeDim(cell_type, cell_dim))
        continue;
      int n_cells = last - first + 1; cgsize_t node_id_list_size;
      cg_ElementDataSize(file_id_, base_id_, zone_id_, section_id,
                         &node_id_list_size);
      auto& section = sections_.emplace_back(file_id_, base_id_, zone_id_,
          new_section_id++, section_name, range_min, n_cells,
          n_boundary_cells, cell_type);
      section.Read();
      range_min += n_cells;
    }
  }
  void ReadSolutions() {
    int n_solutions;
    cg_nsols(file_id_, base_id_, zone_id_, &n_solutions);
    solutions_.reserve(n_solutions);
    for (int sol_id = 1; sol_id <= n_solutions; ++sol_id) {
      char sol_name[33];
      CGNS_ENUMT(GridLocation_t) location;
      cg_sol_info(file_id_, base_id_, zone_id_, sol_id, sol_name, &location);
      auto& solution = solutions_.emplace_back(file_id_, base_id_, zone_id_,
          sol_id, sol_name, location);
      int n_fields;
      cg_nfields(file_id_, base_id_, zone_id_, sol_id, &n_fields);
      for (int field_id = 1; field_id <= n_fields; ++field_id) {
        CGNS_ENUMT(DataType_t) datatype;
        char field_name[33];
        cg_field_info(file_id_, base_id_, zone_id_, sol_id,
                      field_id, &datatype, field_name);
        cgsize_t first{1}, last{1};
        if (location == CGNS_ENUMV(Vertex)) {
          last = CountNodes();
        } else if (location == CGNS_ENUMV(CellCenter)) {
          last = CountCells();
        } else {
          assert(false);
        }
        auto name = std::string(field_name);
        solution.fields().emplace(name, Field<Real>(last));
        cg_field_read(file_id_, base_id_, zone_id_, sol_id, field_name,
                      datatype, &first, &last, solution.fields()[name].data());
      }
    }
  }
  void Write() {
    int zone_id;
    auto node_size = static_cast<cgsize_t>(CountNodes());
    cgsize_t zone_size[3] = {node_size, cell_size_, 0};
    cg_zone_write(file_id_, base_id_, name_.c_str(), zone_size,
                  CGNS_ENUMV(Unstructured), &zone_id);
    assert(zone_id == zone_id_);
    coordinates_.Write();
    for (auto& section : sections_) {
      section.Write();
    }
    for (auto& solution : solutions_) {
      solution.Write();
    }
  }
  /*
  void AddSolution(int sol_id, char const* sol_name,
                   CGNS_ENUMT(GridLocation_t) location) {
    solutions_.reserve(sol_id);
    solutions_.emplace_back(sol_name, sol_id, location);
  }
   */

 private:
  std::string name_;
  CoordinatesType coordinates_;
  std::vector<SectionType> sections_;
  std::vector<SolutionType> solutions_;
  cgsize_t cell_size_;
  int file_id_, base_id_, zone_id_;
};

template <class Real>
class Base {
 public:
  using ZoneType = Zone<Real>;
  Base() = default;
  Base(int fid, int bid, char const* name, int cell_dim, int phys_dim)
      : file_id_(fid), base_id_(bid), name_(name),
        cell_dim_(cell_dim), phys_dim_(phys_dim) {
  }
  int id() const {
    return base_id_;
  }
  int GetCellDim() const {
    return cell_dim_;
  }
  int GetPhysDim() const {
    return phys_dim_;
  }
  const std::string& name() const {
    return name_;
  }
  int CountZones() const {
    return zones_.size();
  }
  ZoneType& GetZone(int id) {
    return zones_.at(id-1);
  }
  const ZoneType& GetZone(int id) const {
    return zones_.at(id-1);
  }
  void ReadZones() {
    int n_zones;
    cg_nzones(file_id_, base_id_, &n_zones);
    zones_.reserve(n_zones);
    for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
      char zone_name[33];
      cgsize_t zone_size[3][1];
      cg_zone_read(file_id_, base_id_, zone_id, zone_name, zone_size[0]);
      auto& zone = zones_.emplace_back(file_id_, base_id_, zone_id, zone_name,
          /* n_cells */zone_size[1][0], /* n_nodes */zone_size[0][0]);
      zone.ReadCoordinates();
      zone.ReadAllSections();
      zone.ReadSolutions();
    }
  }
  /*
  void ReadGmshZones(const int& file_id) {
    int n_zones;
    cg_nzones(file_id, base_id_, &n_zones);
    zones_.reserve(n_zones);
    for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
      char zone_name[33]; cgsize_t zone_size[3][1];
      cg_zone_read(file_id, base_id_, zone_id, zone_name, zone_size[0]);
      auto& zone = zones_.emplace_back(zone_name, zone_id, zone_size[0]);
      zone.ReadCoordinates(file_id, base_id_);
      zone.ReadSectionsWithDim(file_id, base_id_, cell_dim_);
    }
  }
   */
  void ReadNodeIdList(const int& file_id) {
    int n_zones;
    cg_nzones(file_id_, base_id_, &n_zones);
    zones_.reserve(n_zones);
    for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
      char zone_name[33];
      cgsize_t zone_size[3][1];
      cg_zone_read(file_id_, base_id_, zone_id, zone_name, zone_size[0]);
      auto& zone = zones_.emplace_back(file_id_, base_id_, zone_id, zone_name,
          /* n_cells */zone_size[1], /* n_nodes */zone_size[0]);
      zone.ReadSectionsWithDim(cell_dim_);
    }
  }
  void Write() {
    int base_id;
    cg_base_write(file_id_, name_.c_str(), cell_dim_, phys_dim_, &base_id);
    assert(base_id == base_id_);
    for (auto& zone : zones_) {
      zone.Write();
    }
  }

 private:
  std::vector<ZoneType> zones_;
  std::string name_;
  int file_id_, base_id_, cell_dim_, phys_dim_;
};

template <class Real>
class File {
 public:
  using BaseType = Base<Real>;
  explicit File(const std::string& name)
      : name_(name) {
  }
  File(const std::string& dir, const std::string& name)
      : name_(dir) {
    if (name_.back() != '/')
      name_.push_back('/');
    name_ += name;
  }
  void ReadBases() {
    if (cg_open(name_.c_str(), CG_MODE_READ, &file_id_)) {
      cg_error_exit();
    }
    bases_.clear();
    int n_bases;
    cg_nbases(file_id_, &n_bases);
    bases_.reserve(n_bases);
    for (int base_id = 1; base_id <= n_bases; ++base_id) {
      char base_name[33];
      int cell_dim{-1}, phys_dim{-1};
      cg_base_read(file_id_, base_id, base_name, &cell_dim, &phys_dim);
      auto& base = bases_.emplace_back(file_id_, base_id, base_name,
          cell_dim, phys_dim);
      base.ReadZones();
    }
    cg_close(file_id_);
  }
  /*
  void OpenWithGmshCells(const std::string& file_name) {
    name_ = file_name;
    if (cg_open(name_.c_str(), CG_MODE_READ, &file_id_)) { cg_error_exit(); }
    ReadGmshBases();
    cg_close(file_id_);
  }
   */
  void ReadNodeIdList() {
    if (cg_open(name_.c_str(), CG_MODE_READ, &file_id_)) {
      cg_error_exit();
    }
    bases_.clear();
    int n_bases;
    cg_nbases(file_id_, &n_bases);
    bases_.reserve(n_bases);
    for (int base_id = 1; base_id <= n_bases; ++base_id) {
      char base_name[33];
      int cell_dim{-1}, phys_dim{-1};
      cg_base_read(file_id_, base_id, base_name, &cell_dim, &phys_dim);
      auto& base = bases_.emplace_back(base_name, base_id, cell_dim, phys_dim);
      base.ReadNodeIdList(file_id_);
    }
    cg_close(file_id_);
  }
  int id() const {
    return file_id_;
  }
  const std::string& name() const {
    return name_;
  }
  int CountBases() const {
    return bases_.size();
  }
  BaseType& GetBase(int id) {
    return bases_.at(id-1);
  }
  const BaseType& GetBase(int id) const {
    return bases_.at(id-1);
  }
  void Write(const std::string& file_name) {
    int file_id;
    if (cg_open(file_name.c_str(), CG_MODE_WRITE, &file_id)) {
      cg_error_exit();
    }
    assert(file_id == file_id_);
    for (auto& base : bases_) {
      base.Write();
    }
    cg_close(file_id);
  }

 private:
  std::vector<BaseType> bases_;
  std::string name_;
  int file_id_{-1};

  /*
  void ReadGmshBases() {
    bases_.clear();
    int n_bases;
    cg_nbases(file_id_, &n_bases);
    bases_.reserve(n_bases);
    for (int base_id = 1; base_id <= n_bases; ++base_id) {
      char base_name[33];
      int cell_dim{-1}, phys_dim{-1};
      cg_base_read(file_id_, base_id, base_name, &cell_dim, &phys_dim);
      auto& base = bases_.emplace_back(base_name, base_id, cell_dim, phys_dim);
      base.ReadGmshZones(file_id_);
    }
  }
   */
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_TYPES_HPP_