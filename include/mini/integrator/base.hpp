//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_BASE_HPP_
#define MINI_INTEGRATOR_BASE_HPP_

#include <iostream>
#include <Eigen/Dense>

template <class Object>
void print(Object&& obj) {
  std::cout << obj << '\n' << std::endl;
}

template <typename Callable, typename Element>
auto Quadrature(Callable&& f_in_local) {
  constexpr auto kCellDim = Element::GetCellDim();
  using Param = Eigen::Array<Scalar, kCellDim, 1>;
  decltype(f_in_local(Param())) sum; sum *= 0;
  for (int i = 0; i < Element::points.size(); ++i) {
    auto weight = Element::weights[i];
    auto xyz_local = Element::points[i];
    auto f_val = f_in_local(xyz_local);
    f_val *= weight;
    sum += f_val;
  }
  return sum;
}

template <typename Callable, typename Element>
auto Integrate(Callable&& f_in_global, Element&& element) {
  auto f_in_local = [&element, &f_in_global](xyz_local) {
    auto xyz_global = element.local_to_global_Dx1(xyz_local);
    auto f_val = f_in_global(xyz_global);
    auto mat_j = element.jacobian(xyz_local);
    auto det_j = Element::CellDim() < Element::GetPhysDim()
        ? (mat_j.transpose() * mat_j).determinant()
        : mat_j.determinant();
    f_val *= std::sqrt(det_j);
    return f_val;
  };
  return Quadrature<Element>(f_in_local);
}
template <typename Callable, typename Element>
Scalar Innerprod(Callable&& f, Callable&& g, Element&& element) {
  return Integrate([&f, &g](Scalar x, Scalar y, Scalar z){
    return f(x, y, z) * g(x, y, z);
  }, element);
}
template <typename Callable, typename Element>
Scalar Norm(Callable&& f, Element&& element) {
  return std::sqrt(Innerprod(f, f, element));
}

class Face {
  MatDx2 jacobian(Scalar x_local, Scalar y_local) {
    return xyz_global_Dx4_ * diff_shape_local_4x2(x_local, y_local);
  }
};
class Cell {

};

#endif  // MINI_INTEGRATOR_BASE_HPP_
