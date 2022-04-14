// Copyright 2022 PEI Weicheng
#ifndef MINI_AIRCRAFT_SOURCE_HPP_
#define MINI_AIRCRAFT_SOURCE_HPP_

#include <algorithm>
#include <vector>

#include "mini/algebra/eigen.hpp"
#include "mini/geometry/frame.hpp"
#include "mini/integrator/line.hpp"
#include "mini/aircraft/rotary.hpp"

namespace mini {
namespace aircraft {

template <typename P, typename Scalar>
class RotorSource : public Rotor<Scalar> {
 public:
  using Part = P;
  using Cell = typename Part::Cell;
  using Face = typename Cell::Face;
  using Coord = typename Cell::Coord;
  using Coeff = typename Cell::Projection::Coeff;

  // TODO(PVC): apply to Part, rather than Cell
  void UpdateCoeff(const Cell &cell, double t_curr, Coeff *coeff) {
    for (auto &blade : this->blades_) {
      Coord p = blade.GetPoint(0.0);
      Coord q = blade.GetPoint(1.0);
      Coord pq = q - p;
      Scalar r_ratio; bool r_found = false;
      Scalar t_ratio; bool t_found = false;
      for (const Face *face : cell.adj_faces_) {
        const auto &gauss = face->gauss();
        // Currently, only triangle is supported.
        assert(gauss.CountVertices() == 3);
        const auto &a = gauss.GetVertex(0);
        const auto &b = gauss.GetVertex(1);
        const auto &c = gauss.GetVertex(2);
        mini::algebra::Matrix<Scalar, 3, 3> mat;
        mat.col(0) = a - p;
        mat.col(1) = b - p;
        mat.col(2) = c - p;
        assert(mat.determinant());  // may fail when p is in the surface of abc
        Coord lambda = mat.fullPivLu().solve(pq);
        if (lambda.minCoeff() >= 0) {
          auto ratio = 1.0 / lambda.sum();
          if (!r_found) {
            r_ratio = ratio;
            r_found = true;
          } else if (!t_found) {
            t_ratio = ratio;
            t_found = true;
          } else {
            // More than two common points are found.
            assert(false);
          }
        }
      }
      if (r_found && t_found) {
        // Integrate along (r)---(t);
        if (r_ratio > t_ratio) {
          std::swap(r_ratio, t_ratio);
        }
        auto r = blade.GetPoint(r_ratio);
        auto t = blade.GetPoint(t_ratio);
        auto line = mini::integrator::Line<Scalar, 4, 1>({r_ratio, t_ratio});
        auto func = [&cell, &blade](const Scalar &ratio){
          auto section = blade.GetSection(ratio);
          auto xyz = section.GetOrigin();
          const auto &proj = cell.projection_;
          auto cv = proj(xyz);  // conservative variable
          auto uvw = cv.momentum() / cv.mass();
          auto force = section.GetForce(cv.mass(), uvw);
          using Mat3xN = mini::algebra::Matrix<Scalar, 3, Cell::N>;
          Mat3xN prod = force * cell.basis_(xyz).transpose();
          return prod;
        };
        auto integral = line.integrate(func);
        integral *= blade.GetSpan();
        coeff->row(0) += integral.row(0);
        coeff->row(1) += integral.row(1);
        coeff->row(2) += integral.row(2);
      }
    }
  }
};

}  // namespace aircraft
}  // namespace mini

#endif  // MINI_AIRCRAFT_SOURCE_HPP_
