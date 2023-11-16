// Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_SPATIAL_SEM_HPP_
#define MINI_SPATIAL_SEM_HPP_

#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "mini/spatial/fem.hpp"

namespace mini {
namespace spatial {
namespace sem {

/**
 * @brief A specialized version of DG using a Lagrange expansion on Gaussian quadrature points. 
 * 
 * @tparam Part 
 */
template <typename Part>
class DiscontinuousGalerkin : public fem::DiscontinuousGalerkin<Part> {
  using Base = fem::DiscontinuousGalerkin<Part>;

 public:
  using Riemann = typename Base::Riemann;
  using Scalar = typename Base::Scalar;
  using Face = typename Base::Face;
  using Cell = typename Base::Cell;
  using Global = typename Base::Global;
  using Projection = typename Base::Projection;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;
  using Temporal = typename Base::Temporal;
  using Column = typename Base::Column;

 protected:
  using GaussOnCell = typename Projection::Gauss;
  using GaussOnLine = typename GaussOnCell::GaussX;
  static constexpr int kLineQ = GaussOnLine::Q;
  static constexpr int kFaceQ = kLineQ * kLineQ;
  std::vector<std::array<int16_t, kFaceQ>> i_node_on_holder_;
  std::vector<std::array<int16_t, kFaceQ>> i_node_on_sharer_;

  static bool AlmostEqual(Global const &x, Global const &y) {
    return (x - y).norm() < 1e-12;
  }

  void MatchGaussianPoints() {
    auto faces = this->part_ptr_->GetLocalFaces();
    i_node_on_holder_.resize(faces.size());
    i_node_on_sharer_.resize(faces.size());
    for (const Face &face : faces) {
      const auto &face_gauss = face.gauss();
      const auto &holder_gauss = face.holder().gauss();
      const auto &sharer_gauss = face.sharer().gauss();
      for (int f = 0, F = face_gauss.CountPoints(); f < F; ++f) {
        auto &p = face_gauss.GetGlobalCoord(f);
        i_node_on_holder_.at(face.id()).at(f) = -1;
        for (int h = 0, H = holder_gauss.CountPoints(); h < H; ++h) {
          if (AlmostEqual(p, holder_gauss.GetGlobalCoord(h))) {
            i_node_on_holder_[face.id()][f] = h;
            break;
          }
        }
        assert(i_node_on_holder_[face.id()][f] >= 0);
        i_node_on_sharer_.at(face.id()).at(f) = -1;
        for (int s = 0, S = sharer_gauss.CountPoints(); s < S; ++s) {
          if (AlmostEqual(p, sharer_gauss.GetGlobalCoord(s))) {
            i_node_on_sharer_[face.id()][f] = s;
            break;
          }
        }
        assert(i_node_on_sharer_[face.id()][f] >= 0);
      }
    }
  }

 public:
  explicit DiscontinuousGalerkin(Part *part_ptr)
      : Base(part_ptr) {
    MatchGaussianPoints();
  }
  DiscontinuousGalerkin(const DiscontinuousGalerkin &) = default;
  DiscontinuousGalerkin &operator=(const DiscontinuousGalerkin &) = default;
  DiscontinuousGalerkin(DiscontinuousGalerkin &&) noexcept = default;
  DiscontinuousGalerkin &operator=(DiscontinuousGalerkin &&) noexcept = default;
  ~DiscontinuousGalerkin() noexcept = default;

 protected:  // override virtual methods defined in Base
  void AddFluxDivergence(Column *residual) const override {
    if (Part::kDegrees > 0) {
      for (const Cell &cell : this->part_ptr_->GetLocalCells()) {
        auto i_cell = cell.id();
        auto *data = residual->data() + this->part_ptr_->GetCellDataOffset(i_cell);
        const auto &gauss = cell.gauss();
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          auto const &flux = cell.GetFluxOnGaussianPoint(q);
          auto const &grad = cell.projection().GetBasisGradientsOnGaussianPoint(q);
          Coeff prod = flux * grad;
          prod *= gauss.GetGlobalWeight(q);
          cell.projection().AddCoeffTo(prod, data);
        }
      }
    }
  }
  void AddFluxOnLocalFaces(Column *residual) const override {
    for (const Face &face : this->part_ptr_->GetLocalFaces()) {
      const auto &gauss = face.gauss();
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      const auto &riemann = face.riemann();
      auto *holder_data = residual->data()
          + this->part_ptr_->GetCellDataOffset(holder.id());
      auto *sharer_data = residual->data()
          + this->part_ptr_->GetCellDataOffset(sharer.id());
      auto &i_node_on_holder = i_node_on_holder_[face.id()];
      auto &i_node_on_sharer = i_node_on_sharer_[face.id()];
      for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
        auto c_holder = i_node_on_holder[f];
        auto c_sharer = i_node_on_sharer[f];
        Value u_holder = holder.projection().GetValueOnGaussianPoint(c_holder);
        Value u_sharer = sharer.projection().GetValueOnGaussianPoint(c_sharer);
        Value flux = riemann.GetFluxUpwind(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(f);
        holder.projection().AddValueTo(-flux, holder_data, c_holder);
        sharer.projection().AddValueTo(flux, sharer_data, c_sharer);
      }
    }
  }
  void AddFluxOnGhostFaces(Column *residual) const override {
    this->Base::AddFluxOnGhostFaces(residual);
  }
  void ApplySolidWall(Column *residual) const override {
    this->Base::ApplySolidWall(residual);
  }
  void ApplySupersonicInlet(Column *residual) const override {
    this->Base::ApplySupersonicInlet(residual);
  }
  void ApplySupersonicOutlet(Column *residual) const override {
    this->Base::ApplySupersonicOutlet(residual);
  }
  void ApplySubsonicInlet(Column *residual) const override {
    this->Base::ApplySubsonicInlet(residual);
  }
  void ApplySubsonicOutlet(Column *residual) const override {
    this->Base::ApplySubsonicOutlet(residual);
  }
  void ApplySmartBoundary(Column *residual) const override {
    this->Base::ApplySmartBoundary(residual);
  }
};

}  // namespace sem
}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_SEM_HPP_
