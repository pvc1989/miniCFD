// Copyright 2022 PEI Weicheng
#ifndef MINI_AIRCRAFT_SOURCE_HPP_
#define MINI_AIRCRAFT_SOURCE_HPP_

#include <algorithm>
#include <utility>
#include <vector>

#include "mini/algebra/eigen.hpp"
#include "mini/geometry/frame.hpp"
#include "mini/geometry/intersect.hpp"
#include "mini/integrator/line.hpp"
#include "mini/aircraft/rotor.hpp"

namespace mini {
namespace aircraft {

/**
 * @brief A rotorcraft is an aircraft that has multiple rotors.
 * 
 * @tparam P 
 * @tparam Scalar 
 */
template <typename P, typename Scalar>
class Rotorcraft {
 public:
  using Part = P;
  using Cell = typename Part::Cell;
  using Face = typename Cell::Face;
  using Coord = typename Cell::Coord;
  using Coeff = typename Cell::Projection::Coeff;
  using Conservative = typename Part::Riemann::Conservative;
  using Rotor = mini::aircraft::Rotor<Scalar>;

 private:
  // TODO(PVC): apply to Part, rather than Cell
  void UpdateCoeff(const Cell &cell, double t_curr, Coeff *coeff, Rotor *rotor) {
    rotor->UpdateAzimuth(t_curr);
    for (auto &blade : rotor->blades_) {
      const Coord &p = blade.P();
      const Coord &q = blade.Q();
      const Coord &pq = blade.PQ();
      Scalar r_ratio; bool r_found = false;
      Scalar t_ratio; bool t_found = false;
      for (const Face *face : cell.adj_faces_) {
        if (r_found && t_found) {
          break;
        }
        const auto &gauss = face->gauss();
        // Currently, only triangle is supported.
        assert(gauss.CountVertices() == 3);
        Coord pa = gauss.GetVertex(0) - p;
        Coord pb = gauss.GetVertex(1) - p;
        Coord pc = gauss.GetVertex(2) - p;
        Scalar ratio = -1.0;
        mini::geometry::Intersect(pa, pb, pc, pq, &ratio);
        if (0 <= ratio && ratio <= 1) {
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
      // If only one common point is found, then either p or q is inside.
      if (r_found && !t_found) {
        t_found = true;
        t_ratio = r_ratio < 0.5 ? 0 : 1;
      }
      if (!r_found && t_found) {
        r_found = true;
        r_ratio = t_ratio < 0.5 ? 0 : 1;
      }
      if (r_found && t_found) {
        // Integrate along (r)---(t);
        if (r_ratio > t_ratio) {
          std::swap(r_ratio, t_ratio);
        }
        auto line = mini::integrator::Line<Scalar, 1, 4>(r_ratio, t_ratio);
        auto func = [&cell, &blade](Scalar ratio){
          auto section = blade.GetSection(ratio);
          auto xyz = section.GetOrigin();
          const auto &proj = cell.projection_;
          auto value = proj(xyz);  // conservative variable
          auto &cv = *reinterpret_cast<Conservative *>(&value);
          auto uvw = cv.momentum() / cv.mass();
          auto force = section.GetForce(cv.mass(), uvw);
          using Mat3xN = mini::algebra::Matrix<Scalar, 3, Cell::N>;
          Mat3xN prod = force * cell.basis_(xyz).transpose();
          return prod;
        };
        auto integral = mini::integrator::Integrate(func, line);
        integral *= blade.GetSpan();
        coeff->row(1) += integral.row(0);
        coeff->row(2) += integral.row(1);
        coeff->row(3) += integral.row(2);
      }
    }
  }

 public:
  void UpdateCoeff(const Cell &cell, double t_curr, Coeff *coeff) {
  }

  Rotorcraft& InstallRotor(const Rotor& rotor) {
    rotors_.emplace_back(rotor);
    return *this;
  }

 protected:
  std::vector<Rotor> rotors_;
};

}  // namespace aircraft
}  // namespace mini

#endif  // MINI_AIRCRAFT_SOURCE_HPP_
