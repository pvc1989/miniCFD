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
#include "mini/mesh/part.hpp"
#include "mini/limiter/weno.hpp"
#include "mini/temporal/rk.hpp"
#include "mini/spatial/fem.hpp"

using Scalar = double;

/* Define the Euler system. */
constexpr int kDimensions = 3;
using Primitive = mini::riemann::euler::Primitives<Scalar, kDimensions>;
using Conservative = mini::riemann::euler::Conservatives<Scalar, kDimensions>;
using Gas = mini::riemann::euler::IdealGas<Scalar, 1.4>;
using Unrotated = mini::riemann::euler::Exact<Gas, kDimensions>;
using Riemann = mini::riemann::rotated::Euler<Unrotated>;

/* Define spatial discretization. */
constexpr int kDegrees = 2;
using Projection = mini::polynomial::Projection<Scalar, kDimensions, kDegrees, 5>;
using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
using Cell = typename Part::Cell;
using Face = typename Part::Face;
using Global = typename Cell::Global;
using Value = typename Cell::Value;
using Coeff = typename Cell::Coeff;

using Limiter = mini::limiter::weno::Eigen<Cell>;

using Spatial = mini::spatial::fem::DGwithLimiterAndSource<Part, Limiter>;

/* Choose the time-stepping scheme. */
constexpr int kOrders = std::min(3, kDegrees + 1);
using Temporal = mini::temporal::RungeKutta<kOrders, Scalar>;

/* Define the types of IC and BCs. */
using IC = Value(*)(const Global &);
using BC = void(*)(const std::string &, Spatial *);

int Main(int argc, char* argv[], IC ic, BC bc);

#endif  // DEMO_EULER_SOURCELESS_HPP_
