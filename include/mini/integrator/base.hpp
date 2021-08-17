//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_BASE_HPP_
#define MINI_INTEGRATOR_BASE_HPP_

#include <cmath>
#include <iostream>

namespace mini {
namespace integrator {

template <class Object>
void print(Object&& obj) {
  std::cout << obj << '\n' << std::endl;
}

template <typename Callable, typename Element>
auto Quadrature(Callable&& f_in_local, Element&& element) {
  using LocalCoord = typename Element::LocalCoord;
  decltype(f_in_local(LocalCoord())) sum; sum *= 0;
  for (int i = 0; i < Element::CountQuadPoints(); ++i) {
    auto f_val = f_in_local(Element::GetCoord(i));
    f_val *= Element::GetWeight(i);
    sum += f_val;
  }
  return sum;
}

template <typename Callable, typename Element>
auto Integrate(Callable&& f_in_global, Element&& element) {
  using LocalCoord = typename Element::LocalCoord;
  auto f_in_local = [&element, &f_in_global](const LocalCoord& xyz_local) {
    auto f_val = f_in_global(element.local_to_global_Dx1(xyz_local));
    auto mat_j = element.jacobian(xyz_local);
    auto det_j = Element::CellDim() < Element::PhysDim()
        ? (mat_j.transpose() * mat_j).determinant()
        : mat_j.determinant();
    f_val *= std::sqrt(det_j);
    return f_val;
  };
  return Quadrature(f_in_local, element);
}

template <typename Callable, typename Element>
auto Innerprod(Callable&& f, Callable&& g, Element&& element) {
  using GlobalCoord = typename Element::GlobalCoord;
  return Integrate([&f, &g](const GlobalCoord& xyz_global){
    return f(xyz_global) * g(xyz_global);
  }, element);
}

template <typename Callable, typename Element>
auto Norm(Callable&& f, Element&& element) {
  return std::sqrt(Innerprod(f, f, element));
}

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_BASE_HPP_
