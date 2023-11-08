//  Copyright 2022 PEI Weicheng
#ifndef DEMO_EULER_SOURCELESS_HPP_
#define DEMO_EULER_SOURCELESS_HPP_

#include <algorithm>
#include <string>

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/eigen.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/limiter/weno.hpp"
#include "mini/solver/rkdg.hpp"

/* Define the Euler system. */
constexpr int kDimensions = 3;
using Primitive = mini::riemann::euler::Primitives<double, kDimensions>;
using Conservative = mini::riemann::euler::Conservatives<double, kDimensions>;
using Gas = mini::riemann::euler::IdealGas<double, 1, 4>;
using Unrotated = mini::riemann::euler::Exact<Gas, kDimensions>;
using Riemann = mini::riemann::rotated::Euler<Unrotated>;

constexpr int kDegrees = 2;
using Projection = mini::polynomial::Projection<double, kDimensions, kDegrees, 5>;
using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
using Cell = typename Part::Cell;
using Face = typename Part::Face;
using Global = typename Cell::Global;
using Value = typename Cell::Value;
using Coeff = typename Cell::Coeff;

using Limiter = mini::limiter::weno::Eigen<Cell>;

/* Choose the time-stepping scheme. */
constexpr int kOrders = std::min(3, kDegrees + 1);
using Solver = RungeKutta<kOrders, Part, Limiter>;

using IC = Value(*)(const Global &);
using BC = void(*)(const std::string &, Solver *);

int Main(int argc, char* argv[], IC ic, BC bc);

#endif  // DEMO_EULER_SOURCELESS_HPP_
