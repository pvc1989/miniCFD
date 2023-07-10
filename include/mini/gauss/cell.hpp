//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_CELL_HPP_
#define MINI_GAUSS_CELL_HPP_

#include <concepts>

#include "mini/lagrange/cell.hpp"
#include "mini/gauss/function.hpp"

namespace mini {
namespace gauss {

/**
 * @brief Abstract numerical integrators on volume elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Cell : public lagrange::Cell<Scalar> {
  using Base = lagrange::Cell<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::LocalCoord;
  using typename Base::GlobalCoord;
  using typename Base::Jacobian;

  virtual int CountQuadraturePoints() const = 0;
  virtual const LocalCoord &GetLocalCoord(int i) const = 0;
  virtual const GlobalCoord &GetGlobalCoord(int i) const = 0;
  virtual Real const &GetLocalWeight(int i) const = 0;
  virtual Real const &GetGlobalWeight(int i) const = 0;
  virtual Real volume() const = 0;

  virtual ~Cell() noexcept = default;
};

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_CELL_HPP_
