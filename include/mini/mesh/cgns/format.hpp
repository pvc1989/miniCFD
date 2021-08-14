// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan
/**
 * This file defines wrappers of APIs and types in CGNS/MLL.
 */
#ifndef MINI_MESH_CGNS_FORMAT_HPP_
#define MINI_MESH_CGNS_FORMAT_HPP_

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "cgnslib.h"

namespace mini {
namespace mesh {
namespace cgns {

template <class Real> class File;
template <class Real> class Base;
template <class Real> class Zone;
template <class Real> class Coordinates;
template <class Real> class Section;
template <class Real> class Solution;

/**
 * Wrapper of the `GridCoordinates_t` type.
 */
template <class Real>
class Coordinates {
 public:  // Constructors:
  explicit Coordinates(Zone<Real> const* zone, int size)
      : zone_(zone), x_(size), y_(size), z_(size), name_("GridCoordinates") {
  }

 public:  // Copy Control:
  Coordinates(Coordinates const &) = default;
  Coordinates& operator=(const Coordinates&) = default;
  Coordinates(Coordinates&&) noexcept = default;
  Coordinates& operator=(Coordinates&&) noexcept = default;
  ~Coordinates() noexcept = default;

 public:  // Accessors:
  File<Real> const& file() const {
    return zone_->file();
  }
  Base<Real> const& base() const {
    return zone_->base();
  }
  Zone<Real> const& zone() const {
    return *zone_;
  }
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
  /**
   * Write coordinates to a given `(file, base, zone)` tuple.
   */
  void Write() const {
    int coord_id;
    auto data_type = std::is_same_v<Real, double> ?
        CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
    cg_coord_write(file().id(), base().id(), zone_->id(),
                   data_type, "CoordinateX", x_.data(), &coord_id);
    cg_coord_write(file().id(), base().id(), zone_->id(),
                   data_type, "CoordinateY", y_.data(), &coord_id);
    cg_coord_write(file().id(), base().id(), zone_->id(),
                   data_type, "CoordinateZ", z_.data(), &coord_id);
  }

 public:  // Mutators:
  std::vector<Real>& x() {
    return x_;
  }
  std::vector<Real>& y() {
    return y_;
  }
  std::vector<Real>& z() {
    return z_;
  }
  /**
   * Read coordinates from a given `(file, base, zone)` tuple.
   */
  void Read() {
    // All id's are 1-based when passing to CGNS/MLL.
    cgsize_t first = 1;
    cgsize_t last = CountNodes();
    auto data_type = std::is_same_v<Real, double> ?
        CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
    cg_coord_read(file().id(), base().id(), zone_->id(), "CoordinateX",
                  data_type, &first, &last, x_.data());
    cg_coord_read(file().id(), base().id(), zone_->id(), "CoordinateY",
                  data_type, &first, &last, y_.data());
    cg_coord_read(file().id(), base().id(), zone_->id(), "CoordinateZ",
                  data_type, &first, &last, z_.data());
  }

 private:  // Data Members:
  std::string name_;
  std::vector<Real> x_, y_, z_;
  Zone<Real> const* zone_{nullptr};
  int id_;
};

/**
 * Wrapper of the `Elements_t` type.
 */
template <class Real>
class Section {
  friend class Zone<Real>;
 public:  // Constructors:
  Section() = default;
  Section(Zone<Real> const* zone, int sid,
          char const* name, cgsize_t first, cgsize_t size,
          int n_boundary_cells, CGNS_ENUMT(ElementType_t) type)
      : zone_{zone},
        section_id_{sid}, name_{name}, first_{first}, size_{size},
        n_boundary_cells_{n_boundary_cells}, type_{type},
        node_id_list_(size * CountNodesByType(type)) {
  }

 public:  // Copy Control:
  Section(const Section&) = delete;
  Section& operator=(const Section&) = delete;
  Section(Section&&) noexcept = default;
  Section& operator=(Section&&) noexcept = default;
  ~Section() noexcept = default;

 public:  // Accessors:
  File<Real> const& file() const {
    return zone_->file();
  }
  Base<Real> const& base() const {
    return zone_->base();
  }
  Zone<Real> const& zone() const {
    return *zone_;
  }
  std::string const& name() const {
    return name_;
  }
  int id() const {
    return section_id_;
  }
  cgsize_t CellIdMin() const { return first_; }
  cgsize_t CellIdMax() const { return first_ + size_ - 1; }
  cgsize_t CountCells() const { return size_; }
  CGNS_ENUMT(ElementType_t) type() const {
    return type_;
  }
  static int CountNodesByType(CGNS_ENUMT(ElementType_t) type) {
    int npe;
    cg_npe(type, &npe);
    return npe;
  }
  int CountNodesByType() const {
    return CountNodesByType(type_);
  }
  static int dim(CGNS_ENUMT(ElementType_t) type) {
    int d;
    switch (type) {
    case CGNS_ENUMV(NODE):
      d = 1; break;
    case CGNS_ENUMV(BAR_2):
      d = 1; break;
    case CGNS_ENUMV(TRI_3):
    case CGNS_ENUMV(QUAD_4):
      d = 2; break;
    case CGNS_ENUMV(TETRA_4):
    case CGNS_ENUMV(PENTA_6):
    case CGNS_ENUMV(HEXA_8):
      d = 3; break;
    default:
      d = -1; break;
    }
    return d;
  }
  int dim() const {
    return dim(type_);
  }
  const cgsize_t* GetNodeIdList() const {
    return node_id_list_.data();
  }
  const cgsize_t* GetNodeIdListByNilBasedRow(cgsize_t row) const {
    return node_id_list_.data() + CountNodesByType() * row;
  }
  const cgsize_t* GetNodeIdListByOneBasedCellId(cgsize_t cell_id) const {
    return GetNodeIdListByNilBasedRow(cell_id - first_);
  }
  /**
   * Write node_id_list_ into a given `(file, base, zone)` tuple.
   */
  void Write() const {
    int section_id;
    cg_section_write(file().id(), base().id(), zone_->id(), name_.c_str(), type_,
        CellIdMin(), CellIdMax(), 0, GetNodeIdList(),
        &section_id);
    assert(section_id == section_id_);
  }

 public:  // Mutators:
  cgsize_t* GetNodeIdList() {
    return node_id_list_.data();
  }
  cgsize_t* GetNodeIdListByNilBasedRow(cgsize_t row) {
    return node_id_list_.data() + CountNodesByType() * row;
  }
  cgsize_t* GetNodeIdListByOneBasedCellId(cgsize_t cell_id) {
    return GetNodeIdListByNilBasedRow(cell_id - first_);
  }
  /**
   * Read node_id_list_ from a given `(file, base, zone)` tuple.
   */
  void Read() {
    cg_elements_read(file().id(), base().id(), zone_->id(), section_id_,
                     GetNodeIdList(), NULL/* int* parent_data */);
  }

 private:  // Data Members:
  std::vector<cgsize_t> node_id_list_;
  std::vector<cgsize_t> start_offset_;
  std::string name_;
  Zone<Real> const* zone_{nullptr};
  cgsize_t first_, size_;
  int section_id_, n_boundary_cells_;
  CGNS_ENUMT(ElementType_t) type_;
};

template <class Real>
class Field {
 public:  // Constructor:
  Field(Solution<Real> const* solution, int fid, char const* name, int size)
      : solution_{solution}, field_id_{fid}, name_{name},  data_(size) {
  }

 public:  // Copy control:

 public:  // Accessors:
  Real const& at(int id) const {
    return data_[id-1];
  }
  int size() const {
    return data_.size();
  }
  std::string const& name() const {
    return name_;
  }
  void Write() const {
    int field_id;
    cg_field_write(solution_->file().id(), solution_->base().id(),
        solution_->zone().id(), solution_->id(), CGNS_ENUMV(RealDouble),
        name_.c_str(), data_.data(), &field_id);
    assert(field_id == field_id_);
  }

 public:  // Mutators:
  Real& at(int id) {
    return data_[id-1];
  }
  Real* data() {
    return data_.data();
  }

 private:
  std::vector<Real> data_;
  std::string name_;
  Solution<Real> const* solution_{nullptr};
  int field_id_;
};

template <class Real>
class Solution {
 public:  // Constructors:
  Solution(Zone<Real> const* zone, int sid, char const* name,
           CGNS_ENUMT(GridLocation_t) location)
      : zone_(zone), sol_id_(sid), name_(name), location_(location) {
  }

 public:  // Copy Control:
  Solution(const Solution&) = delete;
  Solution& operator=(const Solution&) = delete;
  Solution(Solution&&) noexcept = default;
  Solution& operator=(Solution&&) noexcept = default;
  ~Solution() noexcept = default;

 public:  // Accessors:
  File<Real> const& file() const {
    return zone_->file();
  }
  Base<Real> const& base() const {
    return zone_->base();
  }
  Zone<Real> const& zone() const {
    return *zone_;
  }
  int id() const {
    return sol_id_;
  }
  std::string const& name() const {
    return name_;
  }
  bool OnNodes() const {
    return location_ == CGNS_ENUMV(Vertex);
  }
  bool OnCells() const {
    return location_ == CGNS_ENUMV(CellCenter);
  }
  Field<Real> const& GetField(int id) const {
    return fields_[id-1];
  }
  int CountFields() const {
    return fields_.size();
  }
  void Write() const {
    int sol_id;
    cg_sol_write(file().id(), base().id(), zone_->id(),
                 name_.c_str(), location_, &sol_id);
    assert(sol_id == sol_id_);
    for (auto& field : fields_) {
      field.Write();
    }
  }

 public:  // Mutators:
  Field<Real>& GetField(int id) {
    return fields_[id-1];
  }
  Field<Real>& AddField(char const* name) {
    assert(OnNodes() || OnCells());
    int size = OnCells() ? zone_->CountCells() : zone_->CountNodes();
    return fields_.emplace_back(this, fields_.size()+1, name, size);
  }

 private:
  std::vector<Field<Real>> fields_;
  std::string name_;
  Zone<Real> const* zone_{nullptr};
  CGNS_ENUMT(GridLocation_t) location_;
  int sol_id_;
};

template <class Real>
class Zone {
 public:  // Constructors:
  /**
   * @brief Construct a new Zone object
   * 
   * @param base the reference to the parent
   * @param zid the id of this zone
   * @param name the name of this zone
   * @param n_cells the number of highest-dim cells in this zone
   * @param n_nodes the number of nodes in this zone
   */
  Zone(Base<Real> const* base, int zid, char const* name,
       cgsize_t n_cells, cgsize_t n_nodes)
      : base_(base), zone_id_(zid), name_(name), n_cells_(n_cells),
        coordinates_(this, n_nodes) {
  }

 public:  // Copy Control:
  Zone(const Zone&) = delete;
  Zone& operator=(const Zone&) = delete;
  Zone(Zone&&) noexcept = default;
  Zone& operator=(Zone&&) noexcept = default;
  ~Zone() noexcept = default;

 public:  // Accessors:
  File<Real> const& file() const {
    return base_->file();
  }
  Base<Real> const& base() const {
    return *base_;
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
  /**
   * @brief 
   * 
   * @return int 
   */
  int CountAllCells() const {
    int total = 0;
    for (auto& section : sections_) {
      total += section->CountCells();
    }
    return total;
  }
  /**
   * @brief 
   * 
   * @return int 
   */
  int CountCells() const {
    return n_cells_;
  }
  int CountCellsByType(CGNS_ENUMT(ElementType_t) type) const {
    int cell_size{0};
    for (auto& section : sections_) {
      if (section->type_ == type) {
        cell_size += section->CountCells();
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
  const Coordinates<Real>& GetCoordinates() const {
    return coordinates_;
  }
  const Section<Real>& GetSection(int id) const {
    return *(sections_.at(id-1));
  }
  const Solution<Real>& GetSolution(int id) const {
    return *(solutions_.at(id-1));
  }
  void Write(int min_dim = 0) const {
    int zone_id;
    cgsize_t zone_size[3] = {CountNodes(), CountCells(), 0};
    cg_zone_write(file().id(), base().id(), name_.c_str(), zone_size,
                  CGNS_ENUMV(Unstructured), &zone_id);
    assert(zone_id == zone_id_);
    coordinates_.Write();
    for (auto& section : sections_) {
      if (section->dim() >= min_dim)
        section->Write();
    }
    for (auto& solution : solutions_) {
      solution->Write();
    }
  }
  /**
  * Return true if the cell type is supported and consistent with the given dim.
  */
  static bool CheckTypeDim(CGNS_ENUMT(ElementType_t) type, int cell_dim) {
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

 public:  // Mutators:
  Coordinates<Real>& GetCoordinates() {
    return coordinates_;
  }
  Section<Real>& GetSection(int id) {
    return *(sections_.at(id-1));
  }
  Solution<Real>& GetSolution(int id) {
    return *(solutions_.at(id-1));
  }
  void ReadCoordinates() {
    coordinates_.Read();
  }
  void ReadAllSections() {
    int n_sections;
    cg_nsections(file().id(), base().id(), zone_id_, &n_sections);
    sections_.reserve(n_sections);
    for (int section_id = 1; section_id <= n_sections; ++section_id) {
      char section_name[33];
      CGNS_ENUMT(ElementType_t) cell_type;
      cgsize_t first, last;
      int n_boundary_cells, parent_flag;
      cg_section_read(file().id(), base().id(), zone_id_, section_id,
                      section_name, &cell_type, &first, &last,
                      &n_boundary_cells, &parent_flag);
      auto& section = sections_.emplace_back(std::make_unique<Section<Real>>(
          this, section_id, section_name, first, /* size = */last - first + 1,
          n_boundary_cells, cell_type));
      section->Read();
    }
    SortSectionsByDim();
  }
  void ReadSolutions() {
    int n_solutions;
    cg_nsols(file().id(), base_->id(), zone_id_, &n_solutions);
    solutions_.reserve(n_solutions);
    for (int sol_id = 1; sol_id <= n_solutions; ++sol_id) {
      char sol_name[33];
      CGNS_ENUMT(GridLocation_t) location;
      cg_sol_info(file().id(), base_->id(), zone_id_,
          sol_id, sol_name, &location);
      auto& solution = solutions_.emplace_back(std::make_unique<Solution<Real>>(
          this, sol_id, sol_name, location));
      int n_fields;
      cg_nfields(file().id(), base_->id(), zone_id_, sol_id, &n_fields);
      for (int field_id = 1; field_id <= n_fields; ++field_id) {
        CGNS_ENUMT(DataType_t) datatype;
        char field_name[33];
        cg_field_info(file().id(), base_->id(), zone_id_, sol_id,
                      field_id, &datatype, field_name);
        int first{1}, last{1};
        if (location == CGNS_ENUMV(Vertex)) {
          last = CountNodes();
        } else if (location == CGNS_ENUMV(CellCenter)) {
          last = CountCells();
        } else {
          assert(false);
        }
        auto& field = solution->AddField(field_name);
        cg_field_read(file().id(), base_->id(), zone_id_, sol_id, field_name,
                      datatype, &first, &last, &(field.at(1)));
      }
    }
  }
  Solution<Real>& AddSolution(char const* sol_name,
      CGNS_ENUMT(GridLocation_t) location) {
    int sol_id = solutions_.size() + 1;
    return *(solutions_.emplace_back(std::make_unique<Solution<Real>>(
        this, sol_id, sol_name, location)));
  }

 private:
  std::string name_;
  Coordinates<Real> coordinates_;
  std::vector<std::unique_ptr<Section<Real>>> sections_;
  std::vector<std::unique_ptr<Solution<Real>>> solutions_;
  Base<Real> const* base_{nullptr};
  cgsize_t n_cells_;
  int zone_id_;

  void SortSectionsByDim() {
    int n = sections_.size();
    auto dim_then_oldid = std::vector<std::pair<int, int>>();
    for (auto &sect : sections_) {
      dim_then_oldid.emplace_back(sect->dim(), sect->id());
    }
    auto order = std::vector<int>(n);
    std::iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&dim_then_oldid](int lid, int rid) {
      return dim_then_oldid[lid].first > dim_then_oldid[rid].first || (
          dim_then_oldid[lid].first == dim_then_oldid[rid].first &&
          dim_then_oldid[lid].second < dim_then_oldid[rid].second);
    });
    auto sorted_sections = std::vector<std::unique_ptr<Section<Real>>>(n);
    for (int i = 0; i < n; ++i) {
      std::swap(sorted_sections[i], sections_[order[i]]);
    }
    std::swap(sorted_sections, sections_);
    int n_cells = 1;
    for (int sect_id = 1; sect_id <= n; ++sect_id) {
      auto& sect = *sections_[sect_id-1];
      sect.section_id_ = sect_id;
      sect.first_ = n_cells;
      n_cells += sect.CountCells();
    }
    assert(n_cells - 1 == CountAllCells());
  }
};

template <class Real>
class Base {
 public:  // Constructors:
  Base(File<Real> const* file, int bid, char const* name,
       int cell_dim, int phys_dim)
      : file_(file), base_id_(bid), name_(name),
        cell_dim_(cell_dim), phys_dim_(phys_dim) {
  }

 public:  // Copy Control:
  Base(const Base&) = delete;
  Base& operator=(const Base&) = delete;
  Base(Base&&) noexcept = default;
  Base& operator=(Base&&) noexcept = default;
  ~Base() noexcept = default;

 public:  // Accessors:
  File<Real> const& file() const {
    return *file_;
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
  const Zone<Real>& GetZone(int id) const {
    return *(zones_.at(id-1));
  }
  void Write(int min_dim = 0) const {
    int base_id;
    cg_base_write(file_->id(), name_.c_str(), cell_dim_, phys_dim_, &base_id);
    assert(base_id == base_id_);
    for (auto& zone : zones_) {
      zone->Write(min_dim);
    }
  }

 public:  // Mutators:
  Zone<Real>& GetZone(int id) {
    return *(zones_.at(id-1));
  }
  void ReadZones() {
    int n_zones;
    cg_nzones(file().id(), base_id_, &n_zones);
    zones_.reserve(n_zones);
    for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
      char zone_name[33];
      cgsize_t zone_size[3][1];
      cg_zone_read(file().id(), base_id_, zone_id, zone_name, zone_size[0]);
      auto& zone = zones_.emplace_back(std::make_unique<Zone<Real>>(
          this, zone_id, zone_name,
          /* n_cells */zone_size[1][0], /* n_nodes */zone_size[0][0]));
      zone->ReadCoordinates();
      zone->ReadAllSections();
      zone->ReadSolutions();
    }
  }
  void ReadNodeIdList(const int& file_id) {
    int n_zones;
    cg_nzones(file().id(), base_id_, &n_zones);
    zones_.reserve(n_zones);
    for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
      char zone_name[33];
      cgsize_t zone_size[3][1];
      cg_zone_read(file().id(), base_id_, zone_id, zone_name, zone_size[0]);
      auto& zone = zones_.emplace_back(std::make_unique<Zone<Real>>(
          this, zone_id, zone_name,
          /* n_cells */zone_size[1][0], /* n_nodes */zone_size[0][0]));
      zone->ReadSectionsWithDim(cell_dim_);
    }
  }

 private:
  std::vector<std::unique_ptr<Zone<Real>>> zones_;
  std::string name_;
  File<Real> const* file_{nullptr};
  int base_id_, cell_dim_, phys_dim_;
};

template <class Real>
class File {
 public:  // Constructors:
  explicit File(const std::string& name)
      : name_(name) {
  }
  File(const std::string& dir, const std::string& name)
      : name_(dir) {
    if (name_.back() != '/')
      name_.push_back('/');
    name_ += name;
  }

 public:  // Copy Control:
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  File(File&&) noexcept = default;
  File& operator=(File&&) noexcept = default;
  ~File() noexcept = default;

 public:  // Accessors:
  int id() const {
    return file_id_;
  }
  std::string const& name() const {
    return name_;
  }
  int CountBases() const {
    return bases_.size();
  }
  Base<Real> const& GetBase(int id) const {
    return *(bases_.at(id-1));
  }

 public:  // Mutators:
  Base<Real>& GetBase(int id) {
    return *(bases_.at(id-1));
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
      auto& base = bases_.emplace_back(std::make_unique<Base<Real>>(
          this, base_id, base_name, cell_dim, phys_dim));
      base->ReadZones();
    }
    cg_close(file_id_);
  }
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
      int cell_dim, phys_dim;
      cg_base_read(file_id_, base_id, base_name, &cell_dim, &phys_dim);
      auto& base = bases_.emplace_back(std::make_unique<Base<Real>>(
          this, base_id, base_name, cell_dim, phys_dim));
      base->ReadNodeIdList(file_id_);
    }
    cg_close(file_id_);
  }
  void Write(const std::string& file_name, int min_dim = 0) {
    name_ = file_name;
    if (cg_open(file_name.c_str(), CG_MODE_WRITE, &file_id_)) {
      cg_error_exit();
    }
    for (auto& base : bases_) {
      base->Write(min_dim);
    }
    cg_close(file_id_);
  }

 private:
  std::vector<std::unique_ptr<Base<Real>>> bases_;
  std::string name_;
  int file_id_;
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_FORMAT_HPP_
