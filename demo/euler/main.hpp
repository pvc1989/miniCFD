//  Copyright 2022 PEI Weicheng
#ifndef DEMO_EULER_MAIN_HPP_
#define DEMO_EULER_MAIN_HPP_

#include <algorithm>
#include <string>

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/eigen.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/stepper/explicit.hpp"

/* Define the Euler system. */
constexpr int kFunc = 5;
constexpr int kDim = 3;
using Primitive = mini::riemann::euler::PrimitiveTuple<double, kDim>;
using Conservative = mini::riemann::euler::ConservativeTuple<double, kDim>;
using Gas = mini::riemann::euler::IdealGas<double, 1, 4>;
using Unrotated = mini::riemann::euler::Exact<Gas, kDim>;
using Riemann = mini::riemann::rotated::Euler<Unrotated>;

constexpr int kOrder = 2;
using Part = mini::mesh::cgns::Part<cgsize_t, kOrder, Riemann>;
using Cell = typename Part::Cell;
using Face = typename Part::Face;
using Coord = typename Cell::Coord;
using Value = typename Cell::Value;
using Coeff = typename Cell::Coeff;

// using Limiter = mini::polynomial::LazyWeno<Cell>;
using Limiter = mini::polynomial::EigenWeno<Cell>;

/* Choose the time-stepping scheme. */
constexpr int kSteps = std::min(3, kOrder + 1);
using Solver = RungeKutta<kSteps, Part, Limiter>;

using IC = Value(*)(const Coord &);
using BC = void(*)(const std::string &, Solver *);

int Main(int argc, char* argv[], IC ic, BC bc);

#endif  // DEMO_EULER_MAIN_HPP_
