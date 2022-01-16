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

namespace cgns {

template class Part<cgsize_t, 0, mini::riemann::rotated::Riemann>;
template class Part<cgsize_t, 2, mini::riemann::rotated::Riemann>;

}  // namespace cgns
}  // namespace mesh

namespace polynomial {

template class mini::polynomial::LazyWeno<Cell0>;
template class mini::polynomial::LazyWeno<Cell2>;

}  // namespace polynomial
}  // namespace mini

template class RungeKutta<1, mini::mesh::cgns::Part0,
    mini::riemann::rotated::Riemann, mini::polynomial::Limiter0>;
template class RungeKutta<3, mini::mesh::cgns::Part2,
    mini::riemann::rotated::Riemann, mini::polynomial::Limiter2>;
