//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include "rkdg.hpp"

namespace mini {
namespace mesh {
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
template class EigenMatrices<double, IdealGas<double, 1, 4>>;
template class Exact<IdealGas<1, 4, double>, 3>;

}  // namespace euler

namespace rotated {

template class Euler<euler::MyRiemann>;

}  // namespace rotated

}  // namespace riemann
}  // namespace mini

namespace mini {
namespace polynomial {

template class mini::polynomial::EigenWeno<MyCell0, MyEigen>;
template class mini::polynomial::EigenWeno<MyCell2, MyEigen>;

}  // namespace polynomial
}  // namespace mini

template class RungeKutta<1, mini::mesh::cgns::MyPart0,
    mini::riemann::rotated::MyRiemann>;
template class RungeKutta<3, mini::mesh::cgns::MyPart2,
    mini::riemann::rotated::MyRiemann>;
