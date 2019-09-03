// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MODEL_FVM_HPP_
#define MINI_MODEL_FVM_HPP_

#include <string>

namespace mini {
namespace model {

template <class Mesh, class Riemann>
class FVM {
 public:
  explicit FVM(Riemann const& riemann) {
  }
  bool ReadMesh(std::string const& file_name) {
    return true;
  }
  // Mutators:
  template <class Visitor>
  void SetInitialState(Visitor&& visitor) {
  }
  template <class Visitor>
  void SetWallBoundary(Visitor&& visitor) {
  }
  void SetTimeSteps(double start, double stop, int n_steps) {
  }
  // Major computation:
  void Calculate() {
  }
};

}  // namespace model
}  // namespace mini

#endif  // MINI_MODEL_FVM_HPP_
