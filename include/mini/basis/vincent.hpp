//  Copyright 2023 PEI Weicheng
#ifndef MINI_BASIS_VINCENT_HPP_
#define MINI_BASIS_VINCENT_HPP_

#include <concepts>

#include <cassert>
#include <cmath>

namespace mini {
namespace basis {

/**
 * @brief The \f$ g_\mathrm{left}(\xi) \f$ and \f$ g_\mathrm{right}(\xi) \f$ in Vincent's ESFR schemes.
 * 
 * @tparam Scalar The type of scalar variables.
 */
template <std::floating_point Scalar>
class Vincent {
  Scalar c_prev_, c_next_;
  int degree_;

 public:
  static Scalar DiscontinuousGalerkin(int degree) {
    return 1.0;
  }
  static Scalar HuynhLumpingLobatto(int degree) {
    return degree / (2.0 * degree + 1);
  }

  Vincent(int degree, Scalar c_next)
      : c_prev_(1 - c_next), c_next_(c_next), degree_(degree) {
    assert(degree > 0);
  }

  /**
   * @brief Get the value of \f$ g_\mathrm{right}(\xi) = \frac12 \left( \mathrm{P}_{k}(\xi) + c_\mathrm{prev} \mathrm{P}_{k-1}(\xi) + c_\mathrm{next} \mathrm{P}_{k+1}(\xi) \right) \f$.
   * 
   * @param local the value of \f$ \xi \f$
   * @return Scalar the value of \f$ g_\mathrm{right}(\xi) \f$
   */
  Scalar LocalToRightValue(Scalar local) const {
    Scalar value = std::legendre(degree_, local);
    value += c_prev_ * std::legendre(degree_ - 1, local);
    value += c_next_ * std::legendre(degree_ + 1, local);
    value *= 0.5;
    return value;
  }
  /**
   * @brief Get the value of \f$ g_\mathrm{left}(\xi) = g_\mathrm{right}(-\xi) \f$.
   * 
   * @param local the value of \f$ \xi \f$
   * @return Scalar the value of \f$ g_\mathrm{left}(\xi) \f$
   */
  Scalar LocalToLeftValue(Scalar local) const {
    return LocalToRightValue(-local);
  }

};

}  // namespace basis
}  // namespace mini

#endif  // MINI_BASIS_VINCENT_HPP_
