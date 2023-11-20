// Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_SPATIAL_SEM_HPP_
#define MINI_SPATIAL_SEM_HPP_

#include <concepts>
#include <ranges>

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
 public:
  using Base = fem::DiscontinuousGalerkin<Part>;
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

  template <std::ranges::input_range R>
  void MatchGaussianPoints(R &&faces) {
    i_node_on_holder_.resize(i_node_on_holder_.size() + faces.size());
    i_node_on_sharer_.resize(i_node_on_sharer_.size() + faces.size());
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

  template <std::ranges::input_range R>
  void MatchGaussianPointsOnBoundaries(R &&faces) {
    for (const Face &face : faces) {
      assert(i_node_on_holder_.size() == face.id());
      auto &curr_face = i_node_on_holder_.emplace_back();
      const auto &face_gauss = face.gauss();
      const auto &holder_gauss = face.holder().gauss();
      for (int f = 0, F = face_gauss.CountPoints(); f < F; ++f) {
        auto &p = face_gauss.GetGlobalCoord(f);
        curr_face.at(f) = -1;
        for (int h = 0, H = holder_gauss.CountPoints(); h < H; ++h) {
          if (AlmostEqual(p, holder_gauss.GetGlobalCoord(h))) {
            curr_face[f] = h;
            break;
          }
        }
        assert(curr_face[f] >= 0);
      }
    }
  }

 public:
  explicit DiscontinuousGalerkin(Part *part_ptr)
      : Base(part_ptr) {
    MatchGaussianPoints(part_ptr->GetLocalFaces());
    MatchGaussianPoints(part_ptr->GetGhostFaces());
    MatchGaussianPointsOnBoundaries(part_ptr->GetBoundaryFaces());
  }
  DiscontinuousGalerkin(const DiscontinuousGalerkin &) = default;
  DiscontinuousGalerkin &operator=(const DiscontinuousGalerkin &) = default;
  DiscontinuousGalerkin(DiscontinuousGalerkin &&) noexcept = default;
  DiscontinuousGalerkin &operator=(DiscontinuousGalerkin &&) noexcept = default;
  ~DiscontinuousGalerkin() noexcept = default;

  Column GetResidualColumn() const override {
    Column residual = this->Base::GetResidualColumn();
    // divide mass matrix for each cell
    for (const Cell &cell : this->part_ptr_->GetLocalCells()) {
      auto i_cell = cell.id();
      auto *data = residual.data() + this->part_ptr_->GetCellDataOffset(i_cell);
      const auto &gauss = cell.gauss();
      for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
        auto scale = 1.0 / gauss.GetGlobalWeight(q);
        data = cell.projection().ScaleValueAt(scale, data);
      }
      assert(data ==
          residual.data() + this->part_ptr_->GetCellDataOffset(i_cell + 1));
    }
    return residual;
  }

 protected:  // override virtual methods defined in Base
  void AddFluxDivergence(Column *residual) const override {
    if (Part::kDegrees > 0) {
      for (const Cell &cell : this->part_ptr_->GetLocalCells()) {
        auto i_cell = cell.id();
        auto *data = residual->data() + this->part_ptr_->GetCellDataOffset(i_cell);
        const auto &gauss = cell.gauss();
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          auto const &value = cell.projection().GetValueOnGaussianPoint(q);
          auto const &flux = Riemann::GetFluxMatrix(value);
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
    for (const Face &face : this->part_ptr_->GetGhostFaces()) {
      const auto &gauss = face.gauss();
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      const auto &riemann = face.riemann();
      auto *holder_data = residual->data()
          + this->part_ptr_->GetCellDataOffset(holder.id());
      auto &i_node_on_holder = i_node_on_holder_[face.id()];
      auto &i_node_on_sharer = i_node_on_sharer_[face.id()];
      for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
        auto c_holder = i_node_on_holder[f];
        auto c_sharer = i_node_on_sharer[f];
        Value u_holder = holder.projection().GetValueOnGaussianPoint(c_holder);
        Value u_sharer = sharer.projection().GetValueOnGaussianPoint(c_sharer);
        Value flux = riemann.GetFluxUpwind(u_holder, u_sharer);
        flux *= -gauss.GetGlobalWeight(f);
        holder.projection().AddValueTo(flux, holder_data, c_holder);
      }
    }
  }
  void ApplySolidWall(Column *residual) const override {
    for (const auto &name : this->solid_wall_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_holder = holder.projection().GetValueOnGaussianPoint(c_holder);
          Value flux = riemann.GetFluxOnSolidWall(u_holder);
          flux *= -gauss.GetGlobalWeight(f);
          holder.projection().AddValueTo(flux, holder_data, c_holder);
        }
      }
    }
  }
  void ApplySupersonicOutlet(Column *residual) const override {
    for (const auto &name : this->supersonic_outlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_holder = holder.projection().GetValueOnGaussianPoint(c_holder);
          Value flux = riemann.GetFluxOnSupersonicOutlet(u_holder);
          flux *= -gauss.GetGlobalWeight(f);
          holder.projection().AddValueTo(flux, holder_data, c_holder);
        }
      }
    }
  }
  void ApplySupersonicInlet(Column *residual) const override {
    for (auto &[name, func] : this->supersonic_inlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value flux = riemann.GetFluxOnSupersonicInlet(u_given);
          flux *= -gauss.GetGlobalWeight(f);
          holder.projection().AddValueTo(flux, holder_data, c_holder);
        }
      }
    }
  }
  void ApplySubsonicInlet(Column *residual) const override {
    for (auto &[name, func] : this->subsonic_inlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_inner = holder.projection().GetValueOnGaussianPoint(c_holder);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value flux = riemann.GetFluxOnSubsonicInlet(u_inner, u_given);
          flux *= -gauss.GetGlobalWeight(f);
          holder.projection().AddValueTo(flux, holder_data, c_holder);
        }
      }
    }
  }
  void ApplySubsonicOutlet(Column *residual) const override {
    for (auto &[name, func] : this->subsonic_outlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_inner = holder.projection().GetValueOnGaussianPoint(c_holder);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value flux = riemann.GetFluxOnSubsonicOutlet(u_inner, u_given);
          flux *= -gauss.GetGlobalWeight(f);
          holder.projection().AddValueTo(flux, holder_data, c_holder);
        }
      }
    }
  }
  void ApplySmartBoundary(Column *residual) const override {
    for (auto &[name, func] : this->smart_boundary_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_inner = holder.projection().GetValueOnGaussianPoint(c_holder);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value flux = riemann.GetFluxOnSmartBoundary(u_inner, u_given);
          flux *= -gauss.GetGlobalWeight(f);
          holder.projection().AddValueTo(flux, holder_data, c_holder);
        }
      }
    }
  }
};

/**
 * @brief A specialized version of FR using a Lagrange expansion on Gaussian quadrature points. 
 * 
 * @tparam Part 
 */
template <typename Part>
class FluxReconstruction : public spatial::FiniteElement<Part> {
 public:
  using Base = spatial::FiniteElement<Part>;
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

 public:
  explicit FluxReconstruction(Part *part_ptr)
      : Base(part_ptr) {
  }
  FluxReconstruction(const FluxReconstruction &) = default;
  FluxReconstruction &operator=(const FluxReconstruction &) = default;
  FluxReconstruction(FluxReconstruction &&) noexcept = default;
  FluxReconstruction &operator=(FluxReconstruction &&) noexcept = default;
  ~FluxReconstruction() noexcept = default;

 protected:  // override virtual methods defined in Base
  void AddFluxDivergence(Column *residual) const override {
  }
  void AddFluxOnLocalFaces(Column *residual) const override {
  }
  void AddFluxOnGhostFaces(Column *residual) const override {
  }
  void ApplySolidWall(Column *residual) const override {
  }
  void ApplySupersonicInlet(Column *residual) const override {
  }
  void ApplySupersonicOutlet(Column *residual) const override {
  }
  void ApplySubsonicInlet(Column *residual) const override {
  }
  void ApplySubsonicOutlet(Column *residual) const override {
  }
  void ApplySmartBoundary(Column *residual) const override {
  }
};

}  // namespace sem
}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_SEM_HPP_
