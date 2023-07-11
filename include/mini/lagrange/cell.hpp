//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_LAGRANGE_CELL_HPP_
#define MINI_LAGRANGE_CELL_HPP_

#include <concepts>

namespace mini {
namespace lagrange {

/**
 * @brief Abstract coordinate map on volume elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Cell {
  using Mat3x3 = algebra::Matrix<Scalar, 3, 3>;
  using Mat3x1 = algebra::Matrix<Scalar, 3, 1>;

 public:
  using Real = Scalar;
  using LocalCoord = Mat3x1;
  using GlobalCoord = Mat3x1;
  using Jacobian = Mat3x3;

  virtual ~Cell() noexcept = default;
  virtual std::vector<Scalar> LocalToShapeFunctions(Scalar, Scalar, Scalar) const = 0;
  virtual std::vector<LocalCoord> LocalToShapeGradients(Scalar, Scalar, Scalar) const = 0;
  virtual int CountVertices() const = 0;
  virtual int CountNodes() const = 0;
  virtual const LocalCoord &GetLocalCoord(int i) const = 0;
  virtual const GlobalCoord &GetGlobalCoord(int i) const = 0;
  virtual const GlobalCoord &center() const = 0;
  virtual void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes) const = 0;

  static constexpr int CellDim() {
    return 3;
  }
  static constexpr int PhysDim() {
    return 3;
  }

  GlobalCoord LocalToGlobal(Scalar x_local, Scalar y_local, Scalar z_local)
      const {
    auto shapes = LocalToShapeFunctions(x_local, y_local, z_local);
    GlobalCoord sum = GetGlobalCoord(0) * shapes[0];
    for (int i = 1; i < CountNodes(); ++i) {
      sum += GetGlobalCoord(i) * shapes[i];
    }
    return sum;
  }
  GlobalCoord LocalToGlobal(const LocalCoord &xyz) const {
    return LocalToGlobal(xyz[0], xyz[1], xyz[2]);
  }

  Jacobian LocalToJacobian(Scalar x_local, Scalar y_local, Scalar z_local)
      const {
    auto shapes = LocalToShapeGradients(x_local, y_local, z_local);
    Jacobian sum = GetGlobalCoord(0) * shapes[0].transpose();
    for (int i = 1; i < CountNodes(); ++i) {
      sum += GetGlobalCoord(i) * shapes[i].transpose();
    }
    return sum;
  }
  Jacobian LocalToJacobian(const LocalCoord &xyz) const {
    return LocalToJacobian(xyz[0], xyz[1], xyz[2]);
  }

  LocalCoord GlobalToLocal(Scalar x_global, Scalar y_global, Scalar z_global)
      const {
    Mat3x1 xyz_global = {x_global, y_global, z_global};
    auto func = [this, &xyz_global](Mat3x1 const &xyz_local) {
      auto res = LocalToGlobal(xyz_local);
      return res -= xyz_global;
    };
    auto jac = [this](LocalCoord const &xyz_local) {
      return LocalToJacobian(xyz_local);
    };
    Mat3x1 xyz0 = {0, 0, 0};
    return root(func, xyz0, jac);
  }
  LocalCoord GlobalToLocal(const GlobalCoord &xyz) const {
    return GlobalToLocal(xyz[0], xyz[1], xyz[2]);
  }

 private:
  template <typename Callable, typename MatJ>
  static Mat3x1 root(
      Callable &&func, Mat3x1 x, MatJ &&matj, Scalar xtol = 1e-5) {
    Mat3x1 res;
    do {
      res = matj(x).partialPivLu().solve(func(x));
      x -= res;
    } while (res.norm() > xtol);
    return x;
  }

};

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_CELL_HPP_
