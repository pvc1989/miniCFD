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

static const std::map<CGNS_ENUMT(ElementType_t), int> n_vertex_of_type
    {{CGNS_ENUMV(NODE)   , 1},
     {CGNS_ENUMV(BAR_2)  , 2},
     {CGNS_ENUMV(TRI_3)  , 3},
     {CGNS_ENUMV(QUAD_4) , 4},
     {CGNS_ENUMV(TETRA_4), 4},
     {CGNS_ENUMV(HEXA_8) , 8}};

template <class Real>
struct Coordinates {
  Coordinates() = default;
  Coordinates(int size) : x(size), y(size), z(size) {}
  std::vector<Real> x;
  std::vector<Real> y;
  std::vector<Real> z;
};

template <class Real>
struct Section {
  Section(char* sn, int si, int fi, int la, int nb, CGNS_ENUMT(ElementType_t) ty)
      : name(sn), id(si), first(fi), last(la), n_boundary(nb), type(ty),
        elements((last-first+1)*n_vertex_of_type.at(ty)) {}
  std::string name;
  int id, first, last, n_boundary;
  CGNS_ENUMT(ElementType_t) type;
  std::vector<int> elements;   
};

using Field = std::vector<double>;
template <class Real>
struct Solution {
  Solution(char* sn, int si, CGNS_ENUMT(GridLocation_t) lc)
      : name(sn), id(si), location(lc) {}
  std::string name;
  int id;
  CGNS_ENUMT(GridLocation_t) location;
  std::map<std::string, std::unique_ptr<Field>> fields;
};

template <class Real>
class Zone {
 public:
  using CoordinatesType = Coordinates<Real>;
  using SectionType = Section<Real>;
  using SolutionType = Solution<Real>;
  Zone() = default;
  Zone(char* name, int id, int* zone_size)
      : name_(name), zone_id_(id), cell_size_(zone_size[1]),
        coordinates_(zone_size[0]) {}
  int GetId() const {
    return zone_id_;
  }
  std::string GetName() const {
    return name_;
  }
  int GetVertexSize() const {
    return coordinates_.x.size();
  }
  int GetCellSize() const {
    return cell_size_;
  }
  int CountSections() const {
    return sections_.size();
  }
  int CountSolutions() const {
    return solutions_.size();
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
 
  void ReadCoordinates(int file_id, int base_id) {
    int first, last;
    cg_coord_read(file_id, base_id, zone_id_, "CoordinateX",
                  CGNS_ENUMV(RealSingle), &first, &last, coordinates_.x.data());
    cg_coord_read(file_id, base_id, zone_id_, "CoordinateY",
                  CGNS_ENUMV(RealSingle), &first, &last, coordinates_.y.data());
    cg_coord_read(file_id, base_id, zone_id_, "CoordinateZ",
                  CGNS_ENUMV(RealSingle), &first, &last, coordinates_.z.data());
  }
  void ReadElements(int file_id, int base_id) {
    int n_sections;
    cg_nsections(file_id, base_id, zone_id_, &n_sections);
    sections_.reserve(n_sections);
    for (int section_id = 1; section_id <= n_sections; ++section_id) {
      char section_name[33];
      CGNS_ENUMT(ElementType_t) element_type;
      int first, last, n_boundary, parent_flag;
      cg_section_read(file_id, base_id, zone_id_, section_id, section_name,
                      &element_type, &first, &last, &n_boundary, &parent_flag);
      auto& section = sections_.emplace_back(section_name, section_id, first,
                                             last, n_boundary, element_type);
      int parent_data;
      cg_elements_read(file_id, base_id, zone_id_, section_id,
                       section.elements.data(), &parent_data);
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
        auto sol_array = std::make_unique<Field>(last);
        cg_field_read(file_id, base_id, zone_id_, sol_id, field_name,
                      datatype, &first, &last, sol_array.get()->data());
        solution.fields.emplace(std::string(field_name), std::move(sol_array));
      }
    }
  }
 private: 
  int zone_id_;
  int cell_size_;
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
  std::string GetName() const {
    return name_;
  }
  int CountZones() const {
    return zones_.size();
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
      int zone_size[3][1];
      cg_zone_read(file_id, base_id_, zone_id, zone_name, zone_size[0]);
      auto& zone = zones_.emplace_back(zone_name, zone_id, zone_size[0]);
      zone.ReadCoordinates(file_id, base_id_);
      zone.ReadElements(file_id, base_id_);
      zone.ReadSolutions(file_id, base_id_);
    }
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
  int GetId() const {
    return file_id_;
  }
  std::string GetName() const {
    return name_;
  }
  int CountBases() const {
    return bases_.size();
  }
  BaseType& GetBase(int id) {
    return bases_.at(id-1);
  }

 private:
  int file_id_{-1};
  std::string name_;
  std::vector<BaseType> bases_;

  void ReadBases() {
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
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_TREE_HPP_
