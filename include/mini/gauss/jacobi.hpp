//  Copyright 2023 PEI Weicheng
#ifndef MINI_GAUSS_JACOBI_HPP_
#define MINI_GAUSS_JACOBI_HPP_

#include <concepts>

#include <array>

namespace mini {
namespace gauss {

/**
 * @brief Gauss--Jacobi quadrature rules, i.e. \f$ \int_{-1}^{1} f(\xi) (1-\xi)^\alpha (1+\xi)^\beta \,\mathrm{d}\xi \approx \sum_{q=1}^{Q} w_q f(\xi_q) \f$
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam Q  Nnumber of quadrature points.
 * @tparam kAlpha  Power of \f$ (1-\xi) \f$ in the integrand.
 * @tparam kBeta  Power of \f$ (1+\xi) \f$ in the integrand.
 */
template <std::floating_point Scalar, int Q, int kAlpha, int kBeta>
struct Jacobi;

template <std::floating_point Scalar>
struct Jacobi<Scalar, 1, 2, 0> {
  using Array = std::array<Scalar, 1>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return { -0.5 };
  }
  static Array BuildWeights() {
    return { 2.6666666666666665 };
  }
};
template <std::floating_point Scalar>
typename Jacobi<Scalar, 1, 2, 0>::Array const
Jacobi<Scalar, 1, 2, 0>::points =
Jacobi<Scalar, 1, 2, 0>::BuildPoints();
template <std::floating_point Scalar>
typename Jacobi<Scalar, 1, 2, 0>::Array const
Jacobi<Scalar, 1, 2, 0>::weights =
Jacobi<Scalar, 1, 2, 0>::BuildWeights();

template <std::floating_point Scalar>
struct Jacobi<Scalar, 2, 2, 0> {
  using Array = std::array<Scalar, 2>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return {
      -0.754970354689117,
      0.08830368802245062
    };
  }
  static Array BuildWeights() {
    return {
      1.860379610028064,
      0.8062870566386026
    };
  }
};
template <std::floating_point Scalar>
typename Jacobi<Scalar, 2, 2, 0>::Array const
Jacobi<Scalar, 2, 2, 0>::points =
Jacobi<Scalar, 2, 2, 0>::BuildPoints();
template <std::floating_point Scalar>
typename Jacobi<Scalar, 2, 2, 0>::Array const
Jacobi<Scalar, 2, 2, 0>::weights =
Jacobi<Scalar, 2, 2, 0>::BuildWeights();

template <std::floating_point Scalar>
struct Jacobi<Scalar, 3, 2, 0> {
  using Array = std::array<Scalar, 3>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return {
      -0.8540119518537006,
      -0.30599246792329643,
      0.41000441977699675
    };
  }
  static Array BuildWeights() {
    return {
      1.2570908885190917,
      1.169970154078929,
      0.23960562406864572
    };
  }
};
template <std::floating_point Scalar>
typename Jacobi<Scalar, 3, 2, 0>::Array const
Jacobi<Scalar, 3, 2, 0>::points =
Jacobi<Scalar, 3, 2, 0>::BuildPoints();
template <std::floating_point Scalar>
typename Jacobi<Scalar, 3, 2, 0>::Array const
Jacobi<Scalar, 3, 2, 0>::weights =
Jacobi<Scalar, 3, 2, 0>::BuildWeights();

template <std::floating_point Scalar>
struct Jacobi<Scalar, 4, 2, 0> {
  using Array = std::array<Scalar, 4>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return {
      -0.9029989011060054,
      -0.5227985248962753,
      0.03409459020873491,
      0.5917028357935458
    };
  }
  static Array BuildWeights() {
    return {
      0.8871073248902219,
      1.1476703183937156,
      0.5490710973833848,
      0.08281792599934465
    };
  }
};
template <std::floating_point Scalar>
typename Jacobi<Scalar, 4, 2, 0>::Array const
Jacobi<Scalar, 4, 2, 0>::points =
Jacobi<Scalar, 4, 2, 0>::BuildPoints();
template <std::floating_point Scalar>
typename Jacobi<Scalar, 4, 2, 0>::Array const
Jacobi<Scalar, 4, 2, 0>::weights =
Jacobi<Scalar, 4, 2, 0>::BuildWeights();

template <std::floating_point Scalar>
struct Jacobi<Scalar, 5, 2, 0> {
  using Array = std::array<Scalar, 5>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return {
      -0.9308421201635698,
      -0.6530393584566087,
      -0.2202272258689614,
      0.26866694526177365,
      0.7021084258940329
    };
  }
  static Array BuildWeights() {
    return {
      0.6541182742861681,
      1.009591695199291,
      0.7136012897727205,
      0.25644480578369516,
      0.03291060162479203,
    };
  }
};
template <std::floating_point Scalar>
typename Jacobi<Scalar, 5, 2, 0>::Array const
Jacobi<Scalar, 5, 2, 0>::points =
Jacobi<Scalar, 5, 2, 0>::BuildPoints();
template <std::floating_point Scalar>
typename Jacobi<Scalar, 5, 2, 0>::Array const
Jacobi<Scalar, 5, 2, 0>::weights =
Jacobi<Scalar, 5, 2, 0>::BuildWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_JACOBI_HPP_
