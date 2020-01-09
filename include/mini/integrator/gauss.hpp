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
// 2-point version:
template <>
struct GaussPairs<2> {
  static constexpr std::array<double, 2> x{+0.5773502691896257,
                                           -0.5773502691896257};
  static constexpr std::array<double, 2> w{1.0, 1.0};
};
constexpr std::array<double, 2> GaussPairs<2>::x;
constexpr std::array<double, 2> GaussPairs<2>::w;
// 3-point version:
template <>
struct GaussPairs<3> {
  static constexpr std::array<double, 3> x{+0.7745966692414834,
                                            0.0,
                                           -0.7745966692414834};
  static constexpr std::array<double, 3> w{0.5555555555555556,
                                           0.8888888888888888,
                                           0.5555555555555556};
};
constexpr std::array<double, 3> GaussPairs<3>::x;
constexpr std::array<double, 3> GaussPairs<3>::w;
// 4-point version:
template <>
struct GaussPairs<4> {
  static constexpr std::array<double, 4> x{+0.8611363115940526,
                                           +0.3399810435848563,
                                           -0.3399810435848563,
                                           -0.8611363115940526};
  static constexpr std::array<double, 4> w{0.34785484513745385,
                                           0.6521451548625462,
                                           0.6521451548625462,
                                           0.34785484513745385};
};
constexpr std::array<double, 4> GaussPairs<4>::x;
constexpr std::array<double, 4> GaussPairs<4>::w;
// 5-point version:
template <>
struct GaussPairs<5> {
  static constexpr std::array<double, 5> x{+0.906179845938664,
                                           +0.5384693101056831,
                                            0.0,
                                           -0.5384693101056831,
                                           -0.906179845938664};
  static constexpr std::array<double, 5> w{0.23692688505618908,
                                           0.47862867049936647,
                                           0.5688888888888889,
                                           0.47862867049936647,
                                           0.23692688505618908};
};
constexpr std::array<double, 5> GaussPairs<5>::x;
constexpr std::array<double, 5> GaussPairs<5>::w;

template <int kPoints>
class Gauss {
 public:
  template <class Function>
  static auto Integrate(Function&& function) {
    auto result = function(GaussPairs<kPoints>::x[0]);
    result *= GaussPairs<kPoints>::w[0];
    for (int i = 1; i != kPoints; ++i) {
      auto value = function(GaussPairs<kPoints>::x[i]);
      value *= GaussPairs<kPoints>::w[i];
      result += value;
    }
    return result;
  }
};


}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_GAUSS_HPP_
