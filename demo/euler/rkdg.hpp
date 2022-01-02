//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef DEMO_EULER_RKDG_HPP_
#define DEMO_EULER_RKDG_HPP_

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/eigen.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/mesh/part.hpp"
#include "mini/integrator/ode.hpp"

namespace mini {
namespace mesh {
namespace cgns {

extern template class Part<cgsize_t, double, 5, 3, 0>;
extern template class Part<cgsize_t, double, 5, 3, 2>;

using MyPart0 = Part<cgsize_t, double, 5, 3, 0>;
using MyPart2 = Part<cgsize_t, double, 5, 3, 2>;

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

namespace mini {
namespace riemann {
namespace euler {

extern template class IdealGas<double, 1, 4>;
extern template class EigenMatrices<double, IdealGas<double, 1, 4>>;
extern template class Exact<IdealGas<double, 1, 4>, 3>;
using MyRiemann = Exact<IdealGas<double, 1, 4>, 3>;

}  // namespace euler

namespace rotated {

extern template class Euler<euler::MyRiemann>;
using MyRiemann = Euler<euler::MyRiemann>;

}  // namespace rotated

}  // namespace riemann
}  // namespace mini

namespace mini {
namespace polynomial {

using MyGas = riemann::euler::IdealGas<double, 1, 4>;
using MyEigen = riemann::euler::EigenMatrices<double, MyGas>;
using MyCell0 = mesh::cgns::MyPart0::CellType;
using MyCell2 = mesh::cgns::MyPart2::CellType;

extern template class mini::polynomial::EigenWeno<MyCell0, MyEigen>;
extern template class mini::polynomial::EigenWeno<MyCell2, MyEigen>;

}  // namespace polynomial
}  // namespace mini

extern template class RungeKutta<1, mini::mesh::cgns::MyPart0,
    mini::riemann::rotated::MyRiemann>;
extern template class RungeKutta<3, mini::mesh::cgns::MyPart2,
    mini::riemann::rotated::MyRiemann>;

#endif  // DEMO_EULER_RKDG_HPP_
