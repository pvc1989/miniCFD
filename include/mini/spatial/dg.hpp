// Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_SPATIAL_DG_HPP_
#define MINI_SPATIAL_DG_HPP_

#include <cassert>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "mini/mesh/part.hpp"
#include "mini/temporal/ode.hpp"

namespace mini {
namespace spatial {

template <typename P>
class DummySource {
 public:
  using Part = P;
  using Cell = typename Part::Cell;
  using Coeff = typename Cell::Projection::Coeff;

  void UpdateCoeff(const Cell &cell, double t_curr, Coeff *coeff) {
  }
};

template <typename Part, typename Limiter, typename Source = DummySource<Part>>
class DiscontinuousGalerkin : public temporal::System<typename Part::Scalar> {
 public:
  using Riemann = typename Part::Riemann;
  using Scalar = typename Part::Scalar;
  using Face = typename Part::Face;
  using Cell = typename Part::Cell;
  using Global = typename Cell::Global;
  using Projection = typename Cell::Projection;
  using Coeff = typename Projection::Coeff;
  using Value = typename Projection::Value;
  using Temporal = temporal::System<typename Part::Scalar>;
  using Column = typename Temporal::Column;

 protected:
  std::vector<std::string> supersonic_outlet_, solid_wall_;
  using Function = std::function<Value(const Global &, double)>;
  std::unordered_map<std::string, Function> supersonic_inlet__,
      subsonic_inlet_, subsonic_outlet_, smart_boundary_;
  Part *part_ptr_;
  Limiter limiter_;
  Source source_;
  Column residual_;
  double t_curr_;

 public:
  explicit DiscontinuousGalerkin(Part *part_ptr,
      const Limiter &limiter, const Source &source = Source())
      : part_ptr_(part_ptr), limiter_(limiter), source_(source) {
    residual_.resize(part_ptr_->GetCellDataSize());
  }
  DiscontinuousGalerkin(const DiscontinuousGalerkin &) = default;
  DiscontinuousGalerkin &operator=(const DiscontinuousGalerkin &) = default;
  DiscontinuousGalerkin(DiscontinuousGalerkin &&) noexcept = default;
  DiscontinuousGalerkin &operator=(DiscontinuousGalerkin &&) noexcept = default;
  ~DiscontinuousGalerkin() noexcept = default;

 public:  // set BCs
  template <typename Callable>
  void SetSmartBoundary(const std::string &name, Callable &&func) {
    smart_boundary_[name] = func;
  }
  template <typename Callable>
  void SetSupersonicInlet(const std::string &name, Callable &&func) {
    supersonic_inlet__[name] = func;
  }
  template <typename Callable>
  void SetSubsonicInlet(const std::string &name, Callable &&func) {
    subsonic_inlet_[name] = func;
  }
  template <typename Callable>
  void SetSubsonicOutlet(const std::string &name, Callable &&func) {
    subsonic_outlet_[name] = func;
  }
  void SetSolidWall(const std::string &name) {
    solid_wall_.emplace_back(name);
  }
  void SetSupersonicOutlet(const std::string &name) {
    supersonic_outlet_.emplace_back(name);
  }

 public:  // implement pure virtual methods declared in Temporal
  void SetTime(double t_curr) override {
    t_curr_ = t_curr;
  }
  void SetSolutionColumn(Column const &column) override {
    for (Cell *cell_ptr: part_ptr_->GetLocalCellPointers()) {
      auto i_cell = cell_ptr->id();
      Scalar const *data = column.data() + part_ptr_->GetCellDataOffset(i_cell);
      cell_ptr->projection_.GetCoeffFrom(data);
    }
    part_ptr_->Reconstruct(limiter_);
  }
  Column GetSolutionColumn() const override {
    Column column(residual_.size());
    for (const auto &cell : part_ptr_->GetLocalCells()) {
      auto i_cell = cell.id();
      Scalar *data = column.data() + part_ptr_->GetCellDataOffset(i_cell);
      cell.projection_.WriteCoeffTo(data);
    }
    return column;
  }
  Column GetResidualColumn() const override {
    // part_ptr_->ShareGhostCellCoeffs();
    // this->InitializeResidual(*part_ptr_);
    // this->UpdateLocalResidual(*part_ptr_);
    // this->UpdateBoundaryResidual(*part_ptr_);
    // part_ptr_->UpdateGhostCellCoeffs();
    // this->UpdateGhostResidual(*part_ptr_);
    return this->residual_;
  }

 protected:
  void InitializeResidual(const Part &part) {
    residual_.resize(part.CountLocalCells());
    for (auto &coeff : residual_) {
      coeff.setZero();
    }
    // Integrate the source term, if there is any.
    if (!std::is_same_v<Source, DummySource<Part>>) {
      for (const Cell &cell : part.GetLocalCells()) {
        auto &coeff = this->residual_.at(cell.id());
        source_.UpdateCoeff(cell, this->t_curr_, &coeff);
      }
    }
    // Integrate the dot-product of flux and gradient, if there is any.
    if (Part::kDegrees > 0) {
      for (const Cell &cell : part.GetLocalCells()) {
        auto &coeff = this->residual_.at(cell.id());
        const auto &gauss = *(cell.gauss_ptr_);
        auto n = gauss.CountPoints();
        for (int q = 0; q < n; ++q) {
          const auto &xyz = gauss.GetGlobalCoord(q);
          Value cv = cell.GlobalToValue(xyz);
          auto flux = Riemann::GetFluxMatrix(cv);
          auto grad = cell.basis().GetGradValue(xyz);
          Coeff prod = flux * grad.transpose();
          prod *= gauss.GetGlobalWeight(q);
          coeff += prod;
        }
      }
    }
  }
  void UpdateLocalResidual(const Part &part) {
    assert(residual_.size() == part.CountLocalCells());
    part.ForEachConstLocalFace([this](const Face &face){
      const auto &gauss = *(face.gauss_ptr_);
      const auto &holder = *(face.holder_);
      const auto &sharer = *(face.sharer_);
      const auto &riemann = (face.riemann_);
      auto n = gauss.CountPoints();
      for (int q = 0; q < n; ++q) {
        const auto &coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.GlobalToValue(coord);
        Value u_sharer = sharer.GlobalToValue(coord);
        Value flux = riemann.GetFluxUpwind(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        this->residual_.at(holder.id())
            -= flux * holder.basis()(coord).transpose();
        this->residual_.at(sharer.id())
            += flux * sharer.basis()(coord).transpose();
      }
    });
  }
  void UpdateGhostResidual(const Part &part) {
    assert(residual_.size() == part.CountLocalCells());
    part.ForEachConstGhostFace([this](const Face &face){
      const auto &gauss = *(face.gauss_ptr_);
      const auto &holder = *(face.holder_);
      const auto &sharer = *(face.sharer_);
      const auto &riemann = (face.riemann_);
      auto n = gauss.CountPoints();
      for (int q = 0; q < n; ++q) {
        const auto &coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.GlobalToValue(coord);
        Value u_sharer = sharer.GlobalToValue(coord);
        Value flux = riemann.GetFluxUpwind(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        Coeff temp = flux * holder.basis()(coord).transpose();
        this->residual_.at(holder.id()) -= temp;
      }
    });
  }
  void ApplySolidWall(const Part &part) {
    auto visit = [this](const Face &face){
      const auto &gauss = face.gauss();
      const auto &holder = face.holder();
      const auto &riemann = face.riemann();
      auto n = gauss.CountPoints();
      for (int q = 0; q < n; ++q) {
        const auto &coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.GlobalToValue(coord);
        Value flux = riemann.GetFluxOnSolidWall(u_holder);
        flux *= gauss.GetGlobalWeight(q);
        this->residual_.at(holder.id())
            -= flux * holder.basis()(coord).transpose();
      }
    };
    for (const auto &name : solid_wall_) {
      part.ForEachConstBoundaryFace(visit, name);
    }
  }
  void ApplySupersonicOutlet(const Part &part) {
    auto visit = [this](const Face &face){
      const auto &gauss = face.gauss();
      const auto &holder = face.holder();
      const auto &riemann = face.riemann();
      auto n = gauss.CountPoints();
      for (int q = 0; q < n; ++q) {
        const auto &coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.GlobalToValue(coord);
        Value flux = riemann.GetFluxOnSupersonicOutlet(u_holder);
        flux *= gauss.GetGlobalWeight(q);
        this->residual_.at(holder.id())
            -= flux * holder.basis()(coord).transpose();
      }
    };
    for (const auto &name : supersonic_outlet_) {
      part.ForEachConstBoundaryFace(visit, name);
    }
  }
  void ApplySupersonicInlet(const Part &part) {
    for (auto iter = supersonic_inlet__.begin();
        iter != supersonic_inlet__.end(); ++iter) {
      auto visit = [this, iter](const Face &face){
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto n = gauss.CountPoints();
        for (int q = 0; q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_given = iter->second(coord, this->t_curr_);
          Value flux = riemann.GetFluxOnSupersonicInlet(u_given);
          flux *= gauss.GetGlobalWeight(q);
          this->residual_.at(holder.id())
              -= flux * holder.basis()(coord).transpose();
        }
      };
      part.ForEachConstBoundaryFace(visit, iter->first);
    }
  }
  void ApplySubsonicInlet(const Part &part) {
    for (auto iter = subsonic_inlet_.begin(); iter != subsonic_inlet_.end();
        ++iter) {
      auto visit = [this, iter](const Face &face){
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto n = gauss.CountPoints();
        for (int q = 0; q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_inner = holder.GlobalToValue(coord);
          Value u_given = iter->second(coord, this->t_curr_);
          Value flux = riemann.GetFluxOnSubsonicInlet(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(q);
          this->residual_.at(holder.id())
              -= flux * holder.basis()(coord).transpose();
        }
      };
      part.ForEachConstBoundaryFace(visit, iter->first);
    }
  }
  void ApplySubsonicOutlet(const Part &part) {
    for (auto iter = subsonic_outlet_.begin(); iter != subsonic_outlet_.end();
        ++iter) {
      auto visit = [this, iter](const Face &face){
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto n = gauss.CountPoints();
        for (int q = 0; q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_inner = holder.GlobalToValue(coord);
          Value u_given = iter->second(coord, this->t_curr_);
          Value flux = riemann.GetFluxOnSubsonicOutlet(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(q);
          this->residual_.at(holder.id())
              -= flux * holder.basis()(coord).transpose();
        }
      };
      part.ForEachConstBoundaryFace(visit, iter->first);
    }
  }
  void ApplySmartBoundary(const Part &part) {
    for (auto iter = smart_boundary_.begin(); iter != smart_boundary_.end();
        ++iter) {
      auto visit = [this, iter](const Face &face){
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto n = gauss.CountPoints();
        for (int q = 0; q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_inner = holder.GlobalToValue(coord);
          Value u_given = iter->second(coord, this->t_curr_);
          Value flux = riemann.GetFluxOnSmartBoundary(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(q);
          this->residual_.at(holder.id())
              -= flux * holder.basis()(coord).transpose();
        }
      };
      part.ForEachConstBoundaryFace(visit, iter->first);
    }
  }
  void UpdateBoundaryResidual(const Part &part) {
    ApplySolidWall(part);
    ApplySupersonicInlet(part);
    ApplySupersonicOutlet(part);
    ApplySubsonicInlet(part);
    ApplySubsonicOutlet(part);
    ApplySmartBoundary(part);
  }
};

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_DG_HPP_
