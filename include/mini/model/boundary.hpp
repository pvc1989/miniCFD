// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_MODEL_BOUNDARY_HPP_
#define MINI_MODEL_BOUNDARY_HPP_

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mini {
namespace model {

template <class Mesh>
class Manager {
 public:
  // Types:
  using WallType = typename Mesh::WallType;
  using Part = std::vector<WallType*>;
  // Mutators:
  void AddInteriorWall(WallType* wall) {
    interior_walls_.emplace_back(wall);
  }
  void AddBoundaryWall(WallType* wall) {
    boundary_walls_.emplace_back(wall);
  }
  template <class Visitor>
  void SetBoundaryName(std::string const& name, Visitor&& visitor) {
    name_to_part_.emplace(name, std::make_unique<Part>());
    auto& part = name_to_part_[name];
    for (auto& wall : boundary_walls_) {
      if (visitor(*wall)) {
        part->emplace_back(wall);
      }
    }
  }
  void SetInletBoundary(std::string const& name) {
    inlet_parts_.emplace_back(name_to_part_[name].get());
  }
  void SetPeriodicBoundary(std::string const& head, std::string const& tail) {
    periodic_part_pairs_.emplace_back(name_to_part_[head].get(),
                                      name_to_part_[tail].get());
    SetPeriodicBoundary(name_to_part_[head].get(), name_to_part_[tail].get());
  }
  void SetFreeBoundary(std::string const& name) {
    free_parts_.emplace_back(name_to_part_[name].get());
  }
  void SetSolidBoundary(std::string const& name) {
    solid_parts_.emplace_back(name_to_part_[name].get());
  }
  void ClearBoundaryCondition() {
    if (CheckBoundaryConditions()) {
      boundary_walls_.clear();
    } else {
      throw std::length_error("Some `WallType`s do not have BC info.");
    }
  }
  // Iterators:
  template<class Visitor>
  void ForEachInteriorWall(Visitor&& visit) {
    for (auto& wall : interior_walls_) {
      visit(wall);
    }
  }
  template<class Visitor>
  void ForEachPeriodicWall(Visitor&& visit) {
    for (auto& [left, right] : periodic_part_pairs_) {
      for (int i = 0; i < left->size(); i++) {
        visit(left->at(i));
        right->at(i)->data.flux = left->at(i)->data.flux;
      }
    }
  }
  template<class Visitor>
  void ForEachFreeWall(Visitor&& visit) {
    for (auto& part : free_parts_) {
      for (auto& wall : *part) {
        visit(wall);
      }
    }
  }
  template<class Visitor>
  void ForEachSolidWall(Visitor&& visit) {
    for (auto& part : solid_parts_) {
      for (auto& wall : *part) {
        visit(wall);
      }
    }
  }
  template<class Visitor>
  void ForEachInletWall(Visitor&& visit) {
    for (auto& part : inlet_parts_) {
      for (auto& wall : *part) {
        visit(wall);
      }
    }
  }

 private:
  // Data members:
  std::vector<WallType*> interior_walls_;
  std::vector<WallType*> boundary_walls_;
  std::vector<Part*> free_parts_;
  std::vector<Part*> solid_parts_;
  std::vector<Part*> inlet_parts_;
  std::vector<std::pair<Part*, Part*>> periodic_part_pairs_;
  std::unordered_map<std::string, std::unique_ptr<Part>> name_to_part_;
  // Implement details:
  void SetPeriodicBoundary(Part* head, Part* tail) {
    assert(head->size() == tail->size());
    auto cmp = [](WallType* a, WallType* b) {
      auto point_a = a->Center();
      auto point_b = b->Center();
      if (point_a.Y() != point_b.Y()) {
        return point_a.Y() < point_b.Y();
      } else {
        return point_a.X() < point_b.X();
      }
    };
    std::sort(head->begin(), head->end(), cmp);
    std::sort(tail->begin(), tail->end(), cmp);
    for (int i = 0; i < head->size(); i++) {
      SewMatchingWalls(head->at(i), tail->at(i));
    }
  }
  void SewMatchingWalls(WallType* a, WallType* b) {
    auto left___in = a->GetPositiveSide();
    auto right__in = a->GetNegativeSide();
    auto left__out = b->GetPositiveSide();
    auto right_out = b->GetNegativeSide();
    if (left___in == nullptr) {
      if (left__out == nullptr) {
        a->SetPositiveSide(right_out);
        b->SetPositiveSide(right__in);
      } else {
        a->SetPositiveSide(left__out);
        b->SetNegativeSide(right__in);
      }
    } else {
      if (left__out == nullptr) {
        a->SetNegativeSide(right_out);
        b->SetPositiveSide(left___in);
      } else {
        a->SetNegativeSide(left__out);
        b->SetNegativeSide(left___in);
      }
    }
  }
  bool CheckBoundaryConditions() {
    int n = 0;
    for (auto& [name, part] : name_to_part_) {
      n += part->size();
    }
    return n == boundary_walls_.size();
  }
};

}  // namespace model
}  // namespace mini

#endif  // MINI_MODEL_BOUNDARY_HPP_
