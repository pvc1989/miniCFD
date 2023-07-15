//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_FACE_HPP_
#define MINI_GAUSS_FACE_HPP_

#include <concepts>

#include "mini/gauss/element.hpp"
#include "mini/gauss/function.hpp"

namespace mini {
namespace gauss {

/**
 * @brief Abstract numerical integrators on surface elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kDimensions  Dimension of the physical space.
 */
template <std::floating_point Scalar, int kDimensions>
class Face {
  static constexpr int D = kDimensions;

  using Mat2x2 = algebra::Matrix<Scalar, 2, 2>;
  using Mat2x1 = algebra::Matrix<Scalar, 2, 1>;
  using MatDx1 = algebra::Matrix<Scalar, D, 1>;
  using MatDx2 = algebra::Matrix<Scalar, D, 2>;
  using MatDxD = algebra::Matrix<Scalar, D, D>;

 public:
  using Real = Scalar;
  using LocalCoord = Mat2x1;
  using GlobalCoord = MatDx1;

  virtual GlobalCoord LocalToGlobal(const LocalCoord &) const = 0;
  virtual MatDx2 Jacobian(const LocalCoord &) const = 0;
  virtual int CountQuadraturePoints() const = 0;
  virtual int CountCorners() const = 0;
  virtual const GlobalCoord &GetVertex(int i) const = 0;
  virtual const LocalCoord &GetLocalCoord(int i) const = 0;
  virtual const GlobalCoord &GetGlobalCoord(int i) const = 0;
  virtual Real const &GetLocalWeight(int i) const = 0;
  virtual Real const &GetGlobalWeight(int i) const = 0;
  virtual GlobalCoord center() const = 0;
  virtual const MatDxD &GetNormalFrame(int i) const = 0;
  virtual Scalar area() const = 0;

  static constexpr int CellDim() {
    return 2;
  }
  static constexpr int PhysDim() {
    return D;
  }

  virtual ~Face() noexcept = default;
};

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_FACE_HPP_
