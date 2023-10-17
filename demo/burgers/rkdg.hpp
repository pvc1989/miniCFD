//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef DEMO_BURGERS_RKDG_HPP_
#define DEMO_BURGERS_RKDG_HPP_

#include "mini/riemann/rotated/burgers.hpp"
#include "mini/riemann/rotated/single.hpp"
#include "mini/limiter/weno.hpp"
#include "mini/mesh/part.hpp"
#include "mini/mesh/shuffler.hpp"
#include "mini/solver/rkdg.hpp"

namespace mini {

namespace riemann {
namespace rotated {

extern template class Burgers<double, 3>;
using Riemann = Burgers<double, 3>;

}  // namespace rotated
}  // namespace riemann

namespace mesh {

extern template class Shuffler<idx_t, double>;

namespace part {

extern template class Part<cgsize_t, 0, mini::riemann::rotated::Riemann>;
extern template class Part<cgsize_t, 2, mini::riemann::rotated::Riemann>;

using Part0 = Part<cgsize_t, 0, mini::riemann::rotated::Riemann>;
using Part2 = Part<cgsize_t, 2, mini::riemann::rotated::Riemann>;

}  // namespace part
}  // namespace mesh

namespace limiter {
namespace weno {

using Cell0 = mesh::part::Part0::Cell;
using Cell2 = mesh::part::Part2::Cell;

extern template class Lazy<Cell0>;
extern template class Lazy<Cell2>;

using Lazy0 = weno::Lazy<Cell0>;
using Lazy2 = weno::Lazy<Cell2>;

}  // namespace weno
}  // namespace limiter

}  // namespace mini

extern template class RungeKutta<1, mini::mesh::part::Part0,
    mini::limiter::weno::Lazy0>;
extern template class RungeKutta<3, mini::mesh::part::Part2,
    mini::limiter::weno::Lazy2>;

#endif  // DEMO_BURGERS_RKDG_HPP_
