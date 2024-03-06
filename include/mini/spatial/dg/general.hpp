// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_DG_GENERAL_HPP_
#define MINI_SPATIAL_DG_GENERAL_HPP_

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
namespace dg {

template <typename Part>
class General : public spatial::FiniteElement<Part> {
  using Base = spatial::FiniteElement<Part>;

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

 public:
  explicit General(Part *part_ptr)
      : Base(part_ptr) {
  }
  General(const General &) = default;
  General &operator=(const General &) = default;
  General(General &&) noexcept = default;
  General &operator=(General &&) noexcept = default;
  ~General() noexcept = default;

  virtual const char *name() const {
    return "DG::General";
  }

 protected:  // implement pure virtual methods declared in Base
  void AddFluxDivergence(Column *residual) const override {
    // Integrate the dot-product of flux and basis gradient, if there is any.
    if (Part::kDegrees > 0) {
      for (const Cell &cell : this->part_ptr_->GetLocalCells()) {
        Scalar *data = this->AddCellDataOffset(residual, cell.id());
        const auto &gauss = cell.gauss();
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &xyz = gauss.GetGlobalCoord(q);
          Value cv = cell.projection().GlobalToValue(xyz);
          auto flux = Riemann::GetFluxMatrix(cv);
          flux *= gauss.GetGlobalWeight(q);
          auto grad = cell.projection().GlobalToBasisGradients(xyz);
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
      for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
        const auto &coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.GlobalToValue(coord);
        Value u_sharer = sharer.GlobalToValue(coord);
        Value flux = face.riemann(q).GetFluxUpwind(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        Coeff prod = flux * holder.GlobalToBasisValues(coord);
        holder.projection().MinusCoeff(prod, holder_data);
        prod = flux * sharer.GlobalToBasisValues(coord);
        sharer.projection().AddCoeffTo(prod, sharer_data);
      }
    }
  }
  void AddFluxOnGhostFaces(Column *residual) const override {
    for (const Face &face : this->part_ptr_->GetGhostFaces()) {
      const auto &gauss = face.gauss();
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
      for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
        const auto &coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.GlobalToValue(coord);
        Value u_sharer = sharer.GlobalToValue(coord);
        Value flux = face.riemann(q).GetFluxUpwind(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        Coeff prod = flux * holder.GlobalToBasisValues(coord);
        holder.projection().MinusCoeff(prod, holder_data);
      }
    }
  }

 protected:  // virtual methods that might be overriden in subclasses
  void ApplySolidWall(Column *residual) const override {
    for (const auto &name : this->solid_wall_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_holder = holder.GlobalToValue(coord);
          Value flux = face.riemann(q).GetFluxOnSolidWall(u_holder);
          flux *= gauss.GetGlobalWeight(q);
          Coeff prod = flux * holder.GlobalToBasisValues(coord);
          holder.projection().MinusCoeff(prod, holder_data);
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
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_holder = holder.GlobalToValue(coord);
          Value flux = face.riemann(q).GetFluxOnSupersonicOutlet(u_holder);
          flux *= gauss.GetGlobalWeight(q);
          Coeff prod = flux * holder.GlobalToBasisValues(coord);
          holder.projection().MinusCoeff(prod, holder_data);
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
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_given = func(coord, this->t_curr_);
          Value flux = face.riemann(q).GetFluxOnSupersonicInlet(u_given);
          flux *= gauss.GetGlobalWeight(q);
          Coeff prod = flux * holder.GlobalToBasisValues(coord);
          holder.projection().MinusCoeff(prod, holder_data);
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
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_inner = holder.GlobalToValue(coord);
          Value u_given = func(coord, this->t_curr_);
          Value flux = face.riemann(q).GetFluxOnSubsonicInlet(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(q);
          Coeff prod = flux * holder.GlobalToBasisValues(coord);
          holder.projection().MinusCoeff(prod, holder_data);
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
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_inner = holder.GlobalToValue(coord);
          Value u_given = func(coord, this->t_curr_);
          Value flux = face.riemann(q).GetFluxOnSubsonicOutlet(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(q);
          Coeff prod = flux * holder.GlobalToBasisValues(coord);
          holder.projection().MinusCoeff(prod, holder_data);
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
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_inner = holder.GlobalToValue(coord);
          Value u_given = func(coord, this->t_curr_);
          Value flux = face.riemann(q).GetFluxOnSmartBoundary(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(q);
          Coeff prod = flux * holder.GlobalToBasisValues(coord);
          holder.projection().MinusCoeff(prod, holder_data);
        }
      }
    }
  }
};

template <typename Part, typename Limiter, typename Source = DummySource<Part>>
class WithLimiterAndSource : public General<Part> {
  using Base = General<Part>;

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
  Limiter limiter_;
  Source source_;

 public:
  WithLimiterAndSource(Part *part_ptr,
          const Limiter &limiter, const Source &source = Source())
      : Base(part_ptr), limiter_(limiter), source_(source) {
  }
  WithLimiterAndSource(const WithLimiterAndSource &) = default;
  WithLimiterAndSource &operator=(const WithLimiterAndSource &) = default;
  WithLimiterAndSource(WithLimiterAndSource &&) noexcept = default;
  WithLimiterAndSource &operator=(WithLimiterAndSource &&) noexcept = default;
  ~WithLimiterAndSource() noexcept = default;

 public:  // implement pure virtual methods declared in Temporal
  void SetSolutionColumn(Column const &column) override {
    this->Base::SetSolutionColumn(column);
    this->part_ptr_->Reconstruct(limiter_);
  }

  Column GetResidualColumn() const override {
    auto residual = this->Base::GetResidualColumn();
    this->AddSourceIntegral(&residual);
    return residual;
  }

 protected:
  virtual void AddSourceIntegral(Column *residual) const {
    // Integrate the source term, if there is any.
    if (!std::is_same_v<Source, DummySource<Part>>) {
      for (const Cell &cell : this->part_ptr_->GetLocalCells()) {
        Scalar *data = this->AddCellDataOffset(residual, cell.id());
        const_cast<WithLimiterAndSource *>(this)->source_.UpdateCoeff(
            cell, this->t_curr_, data);
      }
    }
  }
};

}  // namespace dg
}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_DG_GENERAL_HPP_
