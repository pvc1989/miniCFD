// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_MODEL_BOUNDARY_HPP_
#define MINI_MODEL_BOUNDARY_HPP_

#include <algorithm>
#include <cmath>
#include <set>
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
  using Wall = typename Mesh::Wall;
  using Walls = std::set<Wall*>;
  using Part = std::vector<Wall*>;
  using Parts = std::unordered_map<std::string, Part>;
  using Names = std::set<std::vector<Wall*>*>;
  using Pairs = std::set<std::pair<Part*, Part*>>;
  //
  void AddInteriorWall(Wall* wall) {
    interior_walls_.emplace(wall);
  }
  void AddBoundaryWall(Wall* wall) {
    boundary_walls_.emplace(wall);
  }
  // Mutators:
  template <class Visitor>
  void SetBoundaryName(std::string const& name, Visitor&& visitor) {
    name_to_part_.emplace(name, Part());
    auto& part = name_to_part_[name];
    for (auto& wall : boundary_walls_) {
      if (visitor(*wall)) {
        part.emplace_back(wall);
      }
    }
  }
  void SetPeriodicBoundary(std::string const& head, std::string const& tail) {
    periodic_boundaries_.emplace(&(name_to_part_[head]),
                                 &(name_to_part_[tail]));
    SetPeriodicBoundary(&(name_to_part_[head]), &(name_to_part_[tail]));
  }
  void SetFreeBoundary(std::string const& name) {
    free_boundaries_.emplace(&(name_to_part_[name]));
  }
  void SetSolidBoundary(std::string const& name) {
    solid_boundaries_.emplace(&(name_to_part_[name]));
  }
  void ClearBoundaryCondition() {
    if (CheckBoundaryConditions()) {
      boundary_walls_.clear();
    } else {
      throw std::length_error("Some `Wall`s do not have BC info.");
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
    for (auto& [left, right] : periodic_boundaries_) {
      for (int i = 0; i < left->size(); i++) {
        visit(left->at(i));
        right->at(i)->data.flux = left->at(i)->data.flux;
      }
    }
  }
  template<class Visitor>
  void ForEachFreeWall(Visitor&& visit) {
    for (auto& part : free_boundaries_) {
      for (auto& wall : *part) {
        visit(wall);
      }
    }
  }
  template<class Visitor>
  void ForEachSolidWall(Visitor&& visit) {
    for (auto& part : solid_boundaries_) {
      for (auto& wall : *part) {
        visit(wall);
      }
    }
  }

 private:
  // Data members:
  Walls interior_walls_;
  Walls boundary_walls_;
  Parts name_to_part_;
  std::set<Part*> free_boundaries_;
  std::set<Part*> solid_boundaries_;
  Pairs periodic_boundaries_;
  // Implement details:
  void SetPeriodicBoundary(Part* head, Part* tail) {
    assert(head->size() == tail->size());
    auto cmp = [](Wall* a, Wall* b) {
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
  void SewMatchingWalls(Wall* a, Wall* b) {
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
      n += part.size();
    }
    return n == boundary_walls_.size();
  }
};

}  // namespace model
}  // namespace mini

#endif  // MINI_MODEL_BOUNDARY_HPP_
