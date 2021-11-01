//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_FACE_HPP_
#define MINI_INTEGRATOR_FACE_HPP_

#include "mini/integrator/function.hpp"

namespace mini {
namespace integrator {

template <typename Scalar, int D>
class Face {
  using Mat2x2 = algebra::Matrix<Scalar, 2, 2>;
  using Mat2x1 = algebra::Matrix<Scalar, 2, 1>;
  using MatDx1 = algebra::Matrix<Scalar, D, 1>;
  using MatDx2 = algebra::Matrix<Scalar, D, 2>;

 public:
  using Real = Scalar;
  using LocalCoord = Mat2x1;
  using GlobalCoord = MatDx1;

  virtual GlobalCoord LocalToGlobal(const LocalCoord&) const = 0;
  virtual MatDx2 Jacobian(const LocalCoord&) const = 0;
  virtual int CountQuadPoints() const = 0;
  virtual const LocalCoord& GetLocalCoord(int i) const = 0;
  virtual const GlobalCoord& GetGlobalCoord(int i) const = 0;
  virtual const Real& GetLocalWeight(int i) const = 0;
  virtual const Real& GetGlobalWeight(int i) const = 0;
  virtual GlobalCoord center() const = 0;

  static constexpr int CellDim() {
    return 2;
  }
  static constexpr int PhysDim() {
    return D;
  }

  virtual ~Face() noexcept = default;
};

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_FACE_HPP_
