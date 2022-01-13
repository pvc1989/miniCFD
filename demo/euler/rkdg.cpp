//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include "rkdg.hpp"

namespace mini {
namespace mesh {

template class Shuffler<idx_t, double>;

namespace cgns {

template class Part<cgsize_t, double, 5, 3, 0>;
template class Part<cgsize_t, double, 5, 3, 2>;

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

namespace mini {
namespace riemann {
namespace euler {

template class IdealGas<double, 1, 4>;
template class EigenMatrices<IdealGas<double, 1, 4>>;
template class Exact<IdealGas<double, 1, 4>, 3>;

}  // namespace euler

namespace rotated {

template class Euler<euler::Riemann>;

}  // namespace rotated

}  // namespace riemann
}  // namespace mini

namespace mini {
namespace polynomial {

template class mini::polynomial::EigenWeno<Cell0, Eigen>;
template class mini::polynomial::EigenWeno<Cell2, Eigen>;

}  // namespace polynomial
}  // namespace mini

template class RungeKutta<1, mini::mesh::cgns::Part0,
    mini::riemann::rotated::Riemann>;
template class RungeKutta<3, mini::mesh::cgns::Part2,
    mini::riemann::rotated::Riemann>;
