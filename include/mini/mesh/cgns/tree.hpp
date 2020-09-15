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

template <class Real>
struct Coordinates {
  Coordinates() = default;
  Coordinates(int size) {
    x = std::vector<Real>(size);
    y = std::vector<Real>(size);
    z = std::vector<Real>(size);
  }
  std::vector<Real> x;
  std::vector<Real> y;
  std::vector<Real> z;
};

template <class Real>
class Zone {
 public:
  using Coordinates = Coordinates<Real>;
  Zone() = default;
  Zone(char* name, int id, int* zone_size)
    : name_(name), zone_id_(id) {
      vertex_size_ = zone_size[0];
      cell_size_ = zone_size[1];
      boundary_size_ = zone_size[2];
      coordinates_ = Coordinates(vertex_size_);
  }
  int GetId() const {
    return zone_id_;
  }
  std::string GetName() const {
    return name_;
  }
  int GetVertexSize() const {
    return vertex_size_;
  }
  int GetCellSize() const {
    return cell_size_;
  }
  Coordinates& GetCoordinates() {
    return coordinates_;
  }
  void ReadGridCoordinates(const int& file_id, const int& base_id) {
    int irmin, irmax;
    cg_coord_read(file_id, base_id, zone_id_, "CoordinateX",
                  CGNS_ENUMV(RealSingle), &irmin, &irmax, coordinates_.x.data());
    cg_coord_read(file_id, base_id, zone_id_, "CoordinateY",
                  CGNS_ENUMV(RealSingle), &irmin, &irmax, coordinates_.y.data());
    cg_coord_read(file_id, base_id, zone_id_, "CoordinateZ",
                  CGNS_ENUMV(RealSingle), &irmin, &irmax, coordinates_.z.data());
  }
  
 private: 
  int zone_id_;
  int vertex_size_;
  int cell_size_;
  int boundary_size_;
  std::string name_;
  Coordinates coordinates_;
};

template <class Real>
class Base {
 public:
  using Zone = Zone<Real>;
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
    return n_zones_;
  }
  Zone& GetZone(int id) {
    return *(zones_.at(id).get());
  }
  void ReadZones(const int& file_id) {
    cg_nzones(file_id, base_id_, &n_zones_);
    for (int zone_id = 1; zone_id <= n_zones_; ++zone_id) {
      char zone_name[33];
      int zone_size[3][1];
      cg_zone_read(file_id, base_id_, zone_id, zone_name, zone_size[0]);
      auto zone_ptr = std::make_unique<Zone>(zone_name, zone_id, zone_size[0]);
      zone_ptr->ReadGridCoordinates(file_id, base_id_);
      // zone_ptr->ReadElements();
      zones_.emplace(zone_id, std::move(zone_ptr));
    }
  }
  
 private: 
  int base_id_;
  int n_zones_{0};
  int cell_dim_;
  int phys_dim_;
  std::string name_;
  std::map<int, std::unique_ptr<Zone>> zones_;
};

template <class Real>
class Tree {
 public:
  // Types:
  using Base = Base<Real>;
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
    return n_bases_;
  }
  Base& GetBase(int id) {
    return *(bases_.at(id).get());
  }

 private:
  int file_id_{-1};
  int n_bases_{0};
  std::string name_;
  std::map<int, std::unique_ptr<Base>> bases_;

  void ReadBases() {
    cg_nbases(file_id_, &n_bases_);
    for (int base_id = 1; base_id <= n_bases_; ++base_id) {
      char base_name[33];
      int cell_dim{-1}, phys_dim{-1};
      cg_base_read(file_id_, base_id, base_name, &cell_dim, &phys_dim);
      auto base_ptr = std::make_unique<Base>(base_name, base_id,
                                             cell_dim, phys_dim);
      base_ptr->ReadZones(file_id_);
      bases_.emplace(base_id, std::move(base_ptr));
    }
  }
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_TREE_HPP_
