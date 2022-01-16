//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef DEMO_EULER_RKDG_HPP_
#define DEMO_EULER_RKDG_HPP_

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/eigen.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/mesh/part.hpp"
#include "mini/mesh/shuffler.hpp"
#include "mini/stepping/explicit.hpp"

namespace mini {

namespace riemann {
namespace euler {
extern template class IdealGas<double, 1, 4>;
extern template class EigenMatrices<IdealGas<double, 1, 4>>;
extern template class Exact<IdealGas<double, 1, 4>, 3>;
using Riemann = Exact<IdealGas<double, 1, 4>, 3>;
}  // namespace euler
namespace rotated {
extern template class Euler<euler::Riemann>;
using Riemann = Euler<euler::Riemann>;
}  // namespace rotated
}  // namespace riemann

namespace mesh {

extern template class Shuffler<idx_t, double>;

namespace cgns {
extern template class Part<cgsize_t, 0, riemann::rotated::Riemann>;
extern template class Part<cgsize_t, 2, riemann::rotated::Riemann>;
using Part0 = Part<cgsize_t, 0, riemann::rotated::Riemann>;
using Part2 = Part<cgsize_t, 2, riemann::rotated::Riemann>;
}  // namespace cgns
}  // namespace mesh

namespace polynomial {
using Gas = riemann::euler::IdealGas<double, 1, 4>;
using Eigen = riemann::euler::EigenMatrices<Gas>;
using Cell0 = mesh::cgns::Part0::Cell;
using Cell2 = mesh::cgns::Part2::Cell;
extern template class mini::polynomial::EigenWeno<Cell0>;
extern template class mini::polynomial::EigenWeno<Cell2>;
using Limiter0 = mini::polynomial::EigenWeno<Cell0>;
using Limiter2 = mini::polynomial::EigenWeno<Cell2>;
}  // namespace polynomial
}  // namespace mini

extern template class RungeKutta<1, mini::mesh::cgns::Part0,
    mini::riemann::rotated::Riemann, mini::polynomial::Limiter0>;
extern template class RungeKutta<3, mini::mesh::cgns::Part2,
    mini::riemann::rotated::Riemann, mini::polynomial::Limiter2>;

#endif  // DEMO_EULER_RKDG_HPP_
