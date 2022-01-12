//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef DEMO_SINGLE_RKDG_HPP_
#define DEMO_SINGLE_RKDG_HPP_

#include "mini/riemann/rotated/burgers.hpp"
#include "mini/riemann/rotated/single.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/mesh/part.hpp"
#include "mini/mesh/shuffler.hpp"
#include "mini/integrator/ode.hpp"

namespace mini {
namespace mesh {

extern template class Shuffler<idx_t, double>;

namespace cgns {

extern template class Part<cgsize_t, double, 1, 3, 0>;
extern template class Part<cgsize_t, double, 1, 3, 2>;

using MyPart0 = Part<cgsize_t, double, 1, 3, 0>;
using MyPart2 = Part<cgsize_t, double, 1, 3, 2>;

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

namespace mini {
namespace riemann {
namespace rotated {

extern template class Single<3>;
extern template class Burgers<3>;

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

namespace mini {
namespace polynomial {

using MyCell0 = mesh::cgns::MyPart0::Cell;
using MyCell2 = mesh::cgns::MyPart2::Cell;

extern template class mini::polynomial::LazyWeno<MyCell0>;
extern template class mini::polynomial::LazyWeno<MyCell2>;

}  // namespace polynomial
}  // namespace mini

extern template class RungeKutta<1, mini::mesh::cgns::MyPart0,
    mini::riemann::rotated::Burgers<3>>;
extern template class RungeKutta<3, mini::mesh::cgns::MyPart2,
    mini::riemann::rotated::Burgers<3>>;

#endif  // DEMO_SINGLE_RKDG_HPP_
