//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_CELL_HPP_
#define MINI_INTEGRATOR_CELL_HPP_

#include "mini/integrator/function.hpp"

namespace mini {
namespace integrator {

template <typename Scalar>
class Cell {
  using Mat3x3 = algebra::Matrix<Scalar, 3, 3>;
  using Mat3x1 = algebra::Matrix<Scalar, 3, 1>;

 public:
  using Real = Scalar;
  using LocalCoord = Mat3x1;
  using GlobalCoord = Mat3x1;

  virtual GlobalCoord LocalToGlobal(const LocalCoord &) const = 0;
  virtual Mat3x3 Jacobian(const LocalCoord &) const = 0;
  virtual int CountQuadraturePoints() const = 0;
  virtual const LocalCoord &GetLocalCoord(int i) const = 0;
  virtual const GlobalCoord &GetGlobalCoord(int i) const = 0;
  virtual Real const &GetLocalWeight(int i) const = 0;
  virtual Real const &GetGlobalWeight(int i) const = 0;
  virtual GlobalCoord center() const = 0;
  virtual Real volume() const = 0;

  static constexpr int CellDim() {
    return 3;
  }
  static constexpr int PhysDim() {
    return 3;
  }

  virtual ~Cell() noexcept = default;
};

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_CELL_HPP_
