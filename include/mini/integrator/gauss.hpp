//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_INTEGRATOR_GAUSS_HPP_
#define MINI_INTEGRATOR_GAUSS_HPP_

#include <array>


namespace mini {
namespace integrator {

template <int kPoints>
struct GaussPairs;
// 1-point version:
template <>
struct GaussPairs<1> {
  static constexpr std::array<double, 1> x{0.0};
  static constexpr std::array<double, 1> w{2.0};
};
constexpr std::array<double, 1> GaussPairs<1>::x;
constexpr std::array<double, 1> GaussPairs<1>::w;

template <int kPoints>
class Gauss {
 public:
  template <class Function>
  static auto Integrate(Function&& function) {
    auto result = function(GaussPairs<kPoints>::x[0]);
    result *= GaussPairs<kPoints>::w[0];
    for (int i = 1; i != kPoints; ++i) {
      auto value = function(GaussPairs<kPoints>::x[i]);
      value += GaussPairs<kPoints>::w[i];
      result += value;
    }
    return result;
  }
};


}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_GAUSS_HPP_
