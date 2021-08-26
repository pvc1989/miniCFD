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

  virtual GlobalCoord local_to_global_Dx1(const LocalCoord&) const = 0;
  virtual Mat3x3 jacobian(const LocalCoord&) const = 0;
  virtual int CountQuadPoints() const = 0;
  virtual const LocalCoord& GetCoord(int i) const = 0;
  virtual const Real&  GetWeight(int i) const = 0;
  virtual GlobalCoord GetCenter() const = 0;

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
