//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include "rkdg.hpp"

namespace mini {

namespace riemann {
namespace rotated {

template class Burgers<double, 3>;

}  // namespace rotated
}  // namespace riemann

namespace mesh {

template class Shuffler<idx_t, double>;

namespace part {

template class Part<cgsize_t, mini::riemann::rotated::Riemann,
    mini::polynomial::Projection<double, 3, 0, 1>>;
template class Part<cgsize_t, mini::riemann::rotated::Riemann,
    mini::polynomial::Projection<double, 3, 2, 1>>;

}  // namespace part
}  // namespace mesh

namespace limiter {
namespace weno {

template class Lazy<Cell0>;
template class Lazy<Cell2>;

}  // namespace weno
}  // namespace limiter

}  // namespace mini

template class RungeKutta<1, mini::mesh::part::Part0,
    mini::limiter::weno::Lazy0>;
template class RungeKutta<3, mini::mesh::part::Part2,
    mini::limiter::weno::Lazy2>;
