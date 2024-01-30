// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_DG_LOBATTO_HPP_
#define MINI_SPATIAL_DG_LOBATTO_HPP_

#include <concepts>
#include <ranges>

#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "mini/spatial/dg/general.hpp"

namespace mini {
namespace spatial {
namespace dg {

/**
 * @brief A specialized version of DG using a Lagrange expansion on Lobatto roots. 
 * 
 * @tparam Part 
 */
template <typename Part>
class Lobatto : public General<Part> {
 public:
  using Base = General<Part>;
  using Riemann = typename Base::Riemann;
  using Scalar = typename Base::Scalar;
  using Face = typename Base::Face;
  using Cell = typename Base::Cell;
  using Global = typename Base::Global;
  using Gauss = typename Base::Gauss;
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

  using FaceCache = std::array<int16_t, kFaceQ>;
  std::vector<FaceCache> i_node_on_holder_;
  std::vector<FaceCache> i_node_on_sharer_;

  template <std::ranges::input_range R, class FaceToCell>
  void MatchGaussianPoints(R &&faces, FaceToCell &&face_to_cell,
      std::vector<FaceCache> *cache) {
    for (const Face &face : faces) {
      assert(cache->size() == face.id());
      auto &curr_face = cache->emplace_back();
      const auto &face_gauss = face.gauss();
      const auto &cell_gauss = face_to_cell(face).gauss();
      for (int f = 0, F = face_gauss.CountPoints(); f < F; ++f) {
        auto &flux_point = face_gauss.GetGlobalCoord(f);
        curr_face.at(f) = -1;
        for (int h = 0, H = cell_gauss.CountPoints(); h < H; ++h) {
          if (Near(flux_point, cell_gauss.GetGlobalCoord(h))) {
            curr_face[f] = h;
            break;
          }
        }
        assert(curr_face[f] >= 0);
      }
    }
  }

  static constexpr bool kLocal = Projection::kLocal;
  static Scalar GetWeight(const Gauss &gauss, int q) requires(kLocal) {
    return gauss.GetLocalWeight(q);
  }
  static Scalar GetWeight(const Gauss &gauss, int q) requires(!kLocal) {
    return gauss.GetGlobalWeight(q);
  }
  using FluxMatrix = typename Riemann::FluxMatrix;
  static FluxMatrix GetWeightedFluxMatrix(const Value &value,
      const Cell &cell, int q) requires(kLocal) {
    auto flux = Riemann::GetFluxMatrix(value);
    flux = cell.projection().GlobalFluxToLocalFlux(flux, q);
    flux *= GetWeight(cell.gauss(), q);
    return flux;
  }
  static FluxMatrix GetWeightedFluxMatrix(const Value &value,
      const Cell &cell, int q) requires(!kLocal) {
    auto flux = Riemann::GetFluxMatrix(value);
    flux *= GetWeight(cell.gauss(), q);
    return flux;
  }

 public:
  explicit Lobatto(Part *part_ptr)
      : Base(part_ptr) {
    auto face_to_holder = [](auto &face) -> auto & { return face.holder(); };
    auto face_to_sharer = [](auto &face) -> auto & { return face.sharer(); };
    auto local_cells = this->part_ptr_->GetLocalFaces();
    MatchGaussianPoints(local_cells, face_to_holder, &i_node_on_holder_);
    MatchGaussianPoints(local_cells, face_to_sharer, &i_node_on_sharer_);
    auto ghost_cells = this->part_ptr_->GetGhostFaces();
    MatchGaussianPoints(ghost_cells, face_to_holder, &i_node_on_holder_);
    MatchGaussianPoints(ghost_cells, face_to_sharer, &i_node_on_sharer_);
    auto boundary_cells = this->part_ptr_->GetBoundaryFaces();
    MatchGaussianPoints(boundary_cells, face_to_holder, &i_node_on_holder_);
  }
  Lobatto(const Lobatto &) = default;
  Lobatto &operator=(const Lobatto &) = default;
  Lobatto(Lobatto &&) noexcept = default;
  Lobatto &operator=(Lobatto &&) noexcept = default;
  ~Lobatto() noexcept = default;

  virtual const char *name() const {
    return "FR::Lobatto";
  }

  Column GetResidualColumn() const override {
    Column residual = this->Base::GetResidualColumn();
    // divide mass matrix for each cell
    for (const Cell &cell : this->part_ptr_->GetLocalCells()) {
      auto i_cell = cell.id();
      Scalar *data = this->AddCellDataOffset(&residual, i_cell);
      const auto &gauss = cell.gauss();
      for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
        auto scale = 1.0 / GetWeight(gauss, q);
        data = cell.projection().ScaleValueAt(scale, data);
      }
      assert(data == residual.data() + residual.size()
          || data == this->AddCellDataOffset(&residual, i_cell + 1));
    }
    return residual;
  }

 protected:  // override virtual methods defined in Base
  void AddFluxDivergence(Column *residual) const override {
    if (Part::kDegrees > 0) {
      for (const Cell &cell : this->part_ptr_->GetLocalCells()) {
        Scalar *data = this->AddCellDataOffset(residual, cell.id());
        const auto &gauss = cell.gauss();
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          auto const &value = cell.projection().GetValue(q);
          auto flux = GetWeightedFluxMatrix(value, cell, q);
          auto const &grad = cell.projection().GetBasisGradients(q);
          Coeff prod = flux * grad;
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
      Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
      Scalar *sharer_data = this->AddCellDataOffset(residual, sharer.id());
      auto &i_node_on_holder = i_node_on_holder_[face.id()];
      auto &i_node_on_sharer = i_node_on_sharer_[face.id()];
      for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
        auto c_holder = i_node_on_holder[f];
        auto c_sharer = i_node_on_sharer[f];
        Value u_holder = holder.projection().GetValue(c_holder);
        Value u_sharer = sharer.projection().GetValue(c_sharer);
        Value flux = face.riemann(f).GetFluxUpwind(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(f);
        holder.projection().MinusValue(flux, holder_data, c_holder);
        sharer.projection().AddValueTo(flux, sharer_data, c_sharer);
      }
    }
  }
  void AddFluxOnGhostFaces(Column *residual) const override {
    for (const Face &face : this->part_ptr_->GetGhostFaces()) {
      const auto &gauss = face.gauss();
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
      auto &i_node_on_holder = i_node_on_holder_[face.id()];
      auto &i_node_on_sharer = i_node_on_sharer_[face.id()];
      for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
        auto c_holder = i_node_on_holder[f];
        auto c_sharer = i_node_on_sharer[f];
        Value u_holder = holder.projection().GetValue(c_holder);
        Value u_sharer = sharer.projection().GetValue(c_sharer);
        Value flux = face.riemann(f).GetFluxUpwind(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(f);
        holder.projection().MinusValue(flux, holder_data, c_holder);
      }
    }
  }
  void ApplySolidWall(Column *residual) const override {
    for (const auto &name : this->solid_wall_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_holder = holder.projection().GetValue(c_holder);
          Value flux = face.riemann(f).GetFluxOnSolidWall(u_holder);
          flux *= gauss.GetGlobalWeight(f);
          holder.projection().MinusValue(flux, holder_data, c_holder);
        }
      }
    }
  }
  void ApplySupersonicOutlet(Column *residual) const override {
    for (const auto &name : this->supersonic_outlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_holder = holder.projection().GetValue(c_holder);
          Value flux = face.riemann(f).GetFluxOnSupersonicOutlet(u_holder);
          flux *= gauss.GetGlobalWeight(f);
          holder.projection().MinusValue(flux, holder_data, c_holder);
        }
      }
    }
  }
  void ApplySupersonicInlet(Column *residual) const override {
    for (auto &[name, func] : this->supersonic_inlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value flux = face.riemann(f).GetFluxOnSupersonicInlet(u_given);
          flux *= gauss.GetGlobalWeight(f);
          holder.projection().MinusValue(flux, holder_data, c_holder);
        }
      }
    }
  }
  void ApplySubsonicInlet(Column *residual) const override {
    for (auto &[name, func] : this->subsonic_inlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_inner = holder.projection().GetValue(c_holder);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value flux = face.riemann(f).GetFluxOnSubsonicInlet(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(f);
          holder.projection().MinusValue(flux, holder_data, c_holder);
        }
      }
    }
  }
  void ApplySubsonicOutlet(Column *residual) const override {
    for (auto &[name, func] : this->subsonic_outlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_inner = holder.projection().GetValue(c_holder);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value flux = face.riemann(f).GetFluxOnSubsonicOutlet(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(f);
          holder.projection().MinusValue(flux, holder_data, c_holder);
        }
      }
    }
  }
  void ApplySmartBoundary(Column *residual) const override {
    for (auto &[name, func] : this->smart_boundary_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_inner = holder.projection().GetValue(c_holder);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value flux = face.riemann(f).GetFluxOnSmartBoundary(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(f);
          holder.projection().MinusValue(flux, holder_data, c_holder);
        }
      }
    }
  }
};

}  // namespace dg
}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_DG_LOBATTO_HPP_
