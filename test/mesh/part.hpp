// Copyright 2024 PEI Weicheng
#ifndef TEST_MESH_PART_HPP_
#define TEST_MESH_PART_HPP_

#include "mini/riemann/concept.hpp"
#include "mini/riemann/rotated/multiple.hpp"
#include "mini/riemann/diffusive/linear.hpp"
#include "mini/riemann/diffusive/direct_dg.hpp"
#include "mini/gauss/lobatto.hpp"

constexpr int kComponents{2}, kDimensions{3}, kDegrees{2};

using Scalar = double;
using Convection = mini::
    riemann::rotated::Multiple<Scalar, kComponents, kDimensions>;
using Diffusion = mini::riemann::diffusive::DirectDG<
    mini::riemann::diffusive::Isotropic<Scalar, kComponents>
>;
using Riemann = mini::riemann::ConvectionDiffusion<Convection, Diffusion>;
using Coord = typename Riemann::Vector;
using Value = typename Riemann::Conservative;

Value func(const Coord& xyz) {
  auto r = std::hypot(xyz[0] - 2, xyz[1] - 0.5);
  return Value(r, 1 - r + (r >= 1));
}

Value moving(const Coord& xyz, double t) {
  auto x = xyz[0], y = xyz[1];
  return Value(x + y, x - y);
}

int n_core, i_core;
double time_begin;

using Gx = mini::gauss::Lobatto<Scalar, kDegrees + 1>;

#endif  // TEST_MESH_PART_HPP_
