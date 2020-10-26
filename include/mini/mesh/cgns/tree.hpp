// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_CGNS_TREE_HPP_
#define MINI_MESH_CGNS_TREE_HPP_

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

inline int CountNodesByType(CGNS_ENUMT(ElementType_t) type) {
  int npe;
  cg_npe(type, &npe);
  return npe;
}
inline bool CheckTypeDim(CGNS_ENUMT(ElementType_t) type, int cell_dim) {
  if (cell_dim == 2) {
    if (type == CGNS_ENUMV(TRI_3) || type == CGNS_ENUMV(QUAD_4) ||
        type == CGNS_ENUMV( MIXED )) return true;
  } else {
    if (type == CGNS_ENUMV(TETRA_4) || type == CGNS_ENUMV(HEXA_8) ||
        type == CGNS_ENUMV( MIXED )) return true;
  }
  return false;
}

template <class Real> 
struct Coordinates {
 public:  // Constructors:
  explicit Coordinates(int size) : x(size), y(size), z(size) {}
 public:  // Copy Control:
  Coordinates(Coordinates const &) = default;
  Coordinates& operator=(const Coordinates&) = default;
  Coordinates(Coordinates&&) noexcept = default;
  Coordinates& operator=(Coordinates&&) noexcept = default;
  ~Coordinates() noexcept = default;
 public:  // Accessors:
  int CountNodes() const { 
    return x.size();
  }
 public:  // Mutators:
  void Read(int file_id, int base_id, int zone_id) {
    // All id's are 1-based when passing to CGNS/MLL.
    cgsize_t first = 1;
    cgsize_t last = CountNodes();
    auto data_type = std::is_same_v<Real, double> ?
        CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
    cg_coord_read(file_id, base_id, zone_id, "CoordinateX",
                  data_type, &first, &last, x.data());
    cg_coord_read(file_id, base_id, zone_id, "CoordinateY",
                  data_type, &first, &last, y.data());
    cg_coord_read(file_id, base_id, zone_id, "CoordinateZ",
                  data_type, &first, &last, z.data());
  }
  void Write(const int& file_id, const int& base_id, const int& zone_id) {
    int coord_id;
    auto data_type = std::is_same_v<Real, double> ?
        CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
    cg_coord_write(file_id, base_id, zone_id, data_type, "CoordinateX", x.data(), &coord_id);
    cg_coord_write(file_id, base_id, zone_id, data_type, "CoordinateY", y.data(), &coord_id);
    cg_coord_write(file_id, base_id, zone_id, data_type, "CoordinateZ", z.data(), &coord_id);
  }
 public:  // Data Members:
  std::vector<Real> x;
  std::vector<Real> y;
  std::vector<Real> z;
};

template <class Real>
struct Section {
 public:  // Constructors:
  Section(char const* name, int id, cgsize_t first, cgsize_t size,
          int n_boundary_cells, CGNS_ENUMT(ElementType_t) type)
      : name_{name}, id_{id}, first_{first}, size_{size},
        n_boundary_cells_{n_boundary_cells}, type_{type},
        connectivity_(size * CountNodesByType(type)) {}
 public:  // Copy Control:
  Section(const Section&) = default;
  Section& operator=(const Section&) = default;
  Section(Section&&) noexcept = default;
  Section& operator=(Section&&) noexcept = default;
  ~Section() noexcept = default;
 public:  // Accessors:
  std::string const& GetName() const { return name_; }
  int GetId() const { return id_; }
  cgsize_t GetOneBasedCellIdMin() const { return first_; }
  cgsize_t GetOneBasedCellIdMax() const { return first_ + size_ - 1; }
  cgsize_t CountCells() const { return size_; }
  cgsize_t* GetConnectivity() { return connectivity_.data(); }
  cgsize_t* GetConnectivityByNilBasedRow(cgsize_t row) {
    return connectivity_.data() + CountNodesByType(type_) * row;
  }
  cgsize_t* GetConnectivityByOneBasedCellId(cgsize_t cell_id) {
    return GetConnectivityByNilBasedRow(cell_id - first_);
  }
  const cgsize_t* GetConnectivity() const { return connectivity_.data(); }
  const cgsize_t* GetConnectivityByOneBasedCellId(cgsize_t cell_id) const {
    return GetConnectivityByNilBasedRow(cell_id - first_);
  }
  const cgsize_t* GetConnectivityByNilBasedRow(cgsize_t row) const {
    return connectivity_.data() + CountNodesByType(type_) * row;
  }
  CGNS_ENUMT(ElementType_t) GetType() const { return type_; }
 public:  // Mutators:
  void Read(int file_id, int base_id, int zone_id) {
    auto section_id = GetId();
    cg_elements_read(file_id, base_id, zone_id, section_id,
                     GetConnectivity(), NULL/* int* parent_data */);
  }
  void Read(int file_id, int base_id, int zone_id, int section_id) {
    cg_elements_read(file_id, base_id, zone_id, section_id, GetConnectivity(),
                          NULL/* int* parent_data */);
  }
  void Write(const int& file_id, const int& base_id, const int& zone_id) {
    int section_id;
    cg_section_write(file_id, base_id, zone_id, name_.c_str(), type_,
      GetOneBasedCellIdMin(), GetOneBasedCellIdMax(), 0, GetConnectivity(),
      &section_id);
  }
 private:  // Data Members:
  std::vector<cgsize_t> connectivity_;
  std::string name_;
  int id_, n_boundary_cells_;
  cgsize_t first_, size_;
  CGNS_ENUMT(ElementType_t) type_;
};

using Field = std::vector<double>;
template <class Real>
struct Solution {
  Solution(char* sn, int si, CGNS_ENUMT(GridLocation_t) lc)
      : name(sn), id(si), location(lc) {}
  void Write(int file_id, int base_id, int zone_id) {
    int sol_id;
    cg_sol_write(file_id, base_id, zone_id, name.c_str(), location, &sol_id);
    for (auto& [field_name, field] : fields) {
      int field_id;
      cg_field_write(file_id, base_id, zone_id, sol_id, CGNS_ENUMV(RealDouble), field_name.c_str(),
        field.data(), &field_id);
    }
  }
  std::string name;
  int id;
  CGNS_ENUMT(GridLocation_t) location;
  std::map<std::string, Field> fields;
};

template <class Real>
class Zone {
 public:
  using CoordinatesType = Coordinates<Real>;
  using SectionType = Section<Real>;
  using SolutionType = Solution<Real>;
  Zone() = default;
  Zone(char* name, int id, cgsize_t* zone_size)
      : name_(name), zone_id_(id), cell_size_(zone_size[1]),
        coordinates_(zone_size[0]) {}
  int GetId() const {
    return zone_id_;
  }
  const std::string& GetName() const {
    return name_;
  }
  int CountNodes() const {
    return coordinates_.x.size();
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
  void ReadCoordinates(int file_id, int base_id) {
    auto zone_id = GetId();
    coordinates_.Read(file_id, base_id, zone_id);
  }
  void ReadSections(int file_id, int base_id) {
    auto zone_id = GetId();
    int n_sections;
    cg_nsections(file_id, base_id, zone_id, &n_sections);
    sections_.reserve(n_sections);
    for (int section_id = 1; section_id <= n_sections; ++section_id) {
      char section_name[33];
      CGNS_ENUMT(ElementType_t) cell_type;
      cgsize_t first, last;
      int n_boundary_cells, parent_flag;
      cg_section_read(file_id, base_id, zone_id, section_id,
                      section_name, &cell_type, &first, &last,
                      &n_boundary_cells, &parent_flag);
      auto& section = sections_.emplace_back(
          section_name, section_id, first, /* size = */last - first + 1,
          n_boundary_cells, cell_type);
      section.Read(file_id, base_id, zone_id);
    }
  }
  void ReadSectionsWithDim(int file_id, int base_id, int cell_dim) {
    auto zone_id = GetId();
    int n_sections;
    cg_nsections(file_id, base_id, zone_id, &n_sections);
    cgsize_t range_min{1}, range_max; int new_section_id{1};
    for (int section_id = 1; section_id <= n_sections; ++section_id) {
      char section_name[33]; CGNS_ENUMT(ElementType_t) cell_type;
      cgsize_t first, last; int n_boundary_cells, parent_flag;
      cg_section_read(file_id, base_id, zone_id, section_id, section_name, &cell_type,
                      &first, &last, &n_boundary_cells, &parent_flag);
      if (!CheckTypeDim(cell_type, cell_dim)) continue;
      int n_cells = last - first + 1; cgsize_t connectivity_size;
      cg_ElementDataSize(file_id, base_id, zone_id, section_id, &connectivity_size);
      range_max = range_min + n_cells - 1;
      auto& section = sections_.emplace_back(section_name, new_section_id++,
        range_min, n_cells, n_boundary_cells, cell_type);
      section.Read(file_id, base_id, zone_id, section_id);
      range_min = range_max + 1;
    }
  }
  void ReadSolutions(int file_id, int base_id) {
    int n_solutions;
    cg_nsols(file_id, base_id, zone_id_, &n_solutions);
    solutions_.reserve(n_solutions);
    for (int sol_id = 1; sol_id <= n_solutions; ++sol_id) {
      char sol_name[33];
      CGNS_ENUMT(GridLocation_t) location;
      cg_sol_info(file_id, base_id, zone_id_, sol_id, sol_name, &location);
      auto& solution = solutions_.emplace_back(sol_name, sol_id, location);
      int n_fields;
      cg_nfields(file_id, base_id, zone_id_, sol_id, &n_fields);
      for (int field_id = 1; field_id <= n_fields; ++field_id) {
        CGNS_ENUMT(DataType_t) datatype;
        char field_name[33];
        cg_field_info(file_id, base_id, zone_id_, sol_id, field_id, &datatype,
                      field_name);
        int first{1}, last{1};
        if (location == CGNS_ENUMV(Vertex)) {
          last = coordinates_.x.size();
        } else if (location == CGNS_ENUMV(CellCenter)) {
          last = cell_size_;
        }
        std::string name = std::string(field_name);
        solution.fields.emplace(name, Field(last));
        cg_field_read(file_id, base_id, zone_id_, sol_id, field_name,
                      datatype, &first, &last, solution.fields[name].data());
      }
    }
  }
  void Write(const int& file_id, const int& base_id) {
    int zone_id; int node_size = (int)coordinates_.x.size();
    cgsize_t zone_size[3] = {node_size, cell_size_, 0};
    cg_zone_write(file_id, base_id, name_.c_str(), zone_size,
                  CGNS_ENUMV(Unstructured), &zone_id);
    coordinates_.Write(file_id, base_id, zone_id);
    for (auto& section : sections_) { section.Write(file_id, base_id, zone_id); }
    for (auto& solution : solutions_) { solution.Write(file_id, base_id, zone_id); }
  }
  void AddSolution(int sol_id, char* sol_name,
                   CGNS_ENUMT(GridLocation_t) location) {
    solutions_.reserve(sol_id);
    solutions_.emplace_back(sol_name, sol_id, location);                 
  }
 private: 
  int zone_id_;
  cgsize_t cell_size_;
  std::string name_;
  CoordinatesType coordinates_;
  std::vector<SectionType> sections_;
  std::vector<SolutionType> solutions_;
};

template <class Real>
class Base {
 public:
  using ZoneType = Zone<Real>;
  Base() = default;
  Base(char* name, int id, int cell_dim, int phys_dim)
    : name_(name), base_id_(id), cell_dim_(cell_dim), phys_dim_(phys_dim) {}
  int GetId() const {
    return base_id_;
  }
  int GetCellDim() const {
    return cell_dim_;
  }
  int GetPhysDim() const {
    return phys_dim_;
  }
  const std::string& GetName() const {
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
  void ReadZones(const int& file_id) {
    int n_zones;
    cg_nzones(file_id, base_id_, &n_zones);
    zones_.reserve(n_zones);
    for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
      char zone_name[33];
      cgsize_t zone_size[3][1];
      cg_zone_read(file_id, base_id_, zone_id, zone_name, zone_size[0]);
      auto& zone = zones_.emplace_back(zone_name, zone_id, zone_size[0]);
      zone.ReadCoordinates(file_id, base_id_);
      zone.ReadSections(file_id, base_id_);
      zone.ReadSolutions(file_id, base_id_);
    }
  }
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
  void ReadConnectivityForMetis(const int& file_id) {
    int n_zones;
    cg_nzones(file_id, base_id_, &n_zones);
    zones_.reserve(n_zones);
    for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
      char zone_name[33]; cgsize_t zone_size[3][1];
      cg_zone_read(file_id, base_id_, zone_id, zone_name, zone_size[0]);
      auto& zone = zones_.emplace_back(zone_name, zone_id, zone_size[0]);
      zone.ReadSectionsWithDim(file_id, base_id_, cell_dim_);
    }
  }
  void Write(const int& file_id) {
    int base_id;
    cg_base_write(file_id, name_.c_str(), cell_dim_, phys_dim_, &base_id);
    for (auto& zone : zones_) { zone.Write(file_id, base_id); }
  }
  
 private: 
  int base_id_;
  int cell_dim_;
  int phys_dim_;
  std::string name_;
  std::vector<ZoneType> zones_;
};

template <class Real>
class Tree {
 public:
  // Types:
  using BaseType = Base<Real>;
  Tree() = default;
  bool OpenFile(const std::string& file_name) {
    name_ = file_name;
    if (cg_open(name_.c_str(), CG_MODE_READ, &file_id_)) {
      cg_error_exit();
      return false;
    }
    ReadBases();
    cg_close(file_id_);
    return true;
  }
  void OpenFileWithGmshCells(const std::string& file_name) {
    name_ = file_name;
    if (cg_open(name_.c_str(), CG_MODE_READ, &file_id_)) { cg_error_exit(); }
    ReadGmshBases();
    cg_close(file_id_);
  }
  void ReadConnectivityFromFile(const std::string& file_name) {
    name_ = file_name;
    if (cg_open(name_.c_str(), CG_MODE_READ, &file_id_)) { cg_error_exit(); }
    ReadConnectivityForMetis();
    cg_close(file_id_);
  }
  void WriteToFile(const std::string& file_name) {
    int file_id;
    if (cg_open(file_name.c_str(),CG_MODE_WRITE,&file_id)) {
      cg_error_exit();
    }
    for (auto& base : bases_) { base.Write(file_id); }
    cg_close(file_id);
  }
  int GetId() const {
    return file_id_;
  }
  const std::string& GetName() const {
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

 private:
  int file_id_{-1};
  std::string name_;
  std::vector<BaseType> bases_;

  void ReadBases() {
    bases_.clear();
    int n_bases;
    cg_nbases(file_id_, &n_bases);
    bases_.reserve(n_bases);
    for (int base_id = 1; base_id <= n_bases; ++base_id) {
      char base_name[33];
      int cell_dim{-1}, phys_dim{-1};
      cg_base_read(file_id_, base_id, base_name, &cell_dim, &phys_dim);
      auto& base = bases_.emplace_back(base_name, base_id, cell_dim, phys_dim);
      base.ReadZones(file_id_);
    }
  }
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
  void ReadConnectivityForMetis() {
    bases_.clear();
    int n_bases;
    cg_nbases(file_id_, &n_bases);
    bases_.reserve(n_bases);
    for (int base_id = 1; base_id <= n_bases; ++base_id) {
      char base_name[33];
      int cell_dim{-1}, phys_dim{-1};
      cg_base_read(file_id_, base_id, base_name, &cell_dim, &phys_dim);
      auto& base = bases_.emplace_back(base_name, base_id, cell_dim, phys_dim);
      base.ReadConnectivityForMetis(file_id_);
    }
  }
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_TREE_HPP_
