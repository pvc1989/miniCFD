//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef DEMO_BURGERS_RKDG_HPP_
#define DEMO_BURGERS_RKDG_HPP_

#include "mini/riemann/rotated/burgers.hpp"
#include "mini/riemann/rotated/single.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/mesh/part.hpp"
#include "mini/mesh/shuffler.hpp"
#include "mini/stepping/explicit.hpp"

namespace mini {

namespace riemann {
namespace rotated {

extern template class Burgers<double, 3>;
using Riemann = Burgers<double, 3>;

}  // namespace rotated
}  // namespace riemann

namespace mesh {

extern template class Shuffler<idx_t, double>;

namespace cgns {

extern template class Part<cgsize_t, 0, mini::riemann::rotated::Riemann>;
extern template class Part<cgsize_t, 2, mini::riemann::rotated::Riemann>;

using Part0 = Part<cgsize_t, 0, mini::riemann::rotated::Riemann>;
using Part2 = Part<cgsize_t, 2, mini::riemann::rotated::Riemann>;

}  // namespace cgns
}  // namespace mesh

namespace polynomial {

using Cell0 = mesh::cgns::Part0::Cell;
using Cell2 = mesh::cgns::Part2::Cell;

extern template class mini::polynomial::LazyWeno<Cell0>;
extern template class mini::polynomial::LazyWeno<Cell2>;

using Limiter0 = mini::polynomial::LazyWeno<Cell0>;
using Limiter2 = mini::polynomial::LazyWeno<Cell2>;

}  // namespace polynomial
}  // namespace mini

extern template class RungeKutta<1, mini::mesh::cgns::Part0,
    mini::polynomial::Limiter0>;
extern template class RungeKutta<3, mini::mesh::cgns::Part2,
    mini::polynomial::Limiter2>;

#endif  // DEMO_BURGERS_RKDG_HPP_
