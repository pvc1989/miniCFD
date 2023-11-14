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

#include "mini/spatial/fem.hpp"

namespace mini {
namespace spatial {

template <typename Part, typename Limiter, typename Source = DummySource<Part>>
class DiscontinuousGalerkin : public FiniteElement<Part> {
  using Base = FiniteElement<Part>;

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
  DiscontinuousGalerkin(Part *part_ptr,
          const Limiter &limiter, const Source &source = Source())
      : Base(part_ptr), limiter_(limiter), source_(source) {
  }
  DiscontinuousGalerkin(const DiscontinuousGalerkin &) = default;
  DiscontinuousGalerkin &operator=(const DiscontinuousGalerkin &) = default;
  DiscontinuousGalerkin(DiscontinuousGalerkin &&) noexcept = default;
  DiscontinuousGalerkin &operator=(DiscontinuousGalerkin &&) noexcept = default;
  ~DiscontinuousGalerkin() noexcept = default;

 public:  // implement pure virtual methods declared in Temporal
  void SetSolutionColumn(Column const &column) override {
    this->Base::SetSolutionColumn(column);
    this->part_ptr_->Reconstruct(limiter_);
  }
};

/**
 * @brief A specialized version of DG using a Lagrange expansion on Gaussian quadrature points. 
 * 
 * @tparam Part 
 */
template <typename Part>
class DGonGaussianPoints : public FiniteElement<Part> {
  using Base = FiniteElement<Part>;

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

 public:
  explicit DGonGaussianPoints(Part *part_ptr)
      : Base(part_ptr) {
  }
  DGonGaussianPoints(const DGonGaussianPoints &) = default;
  DGonGaussianPoints &operator=(const DGonGaussianPoints &) = default;
  DGonGaussianPoints(DGonGaussianPoints &&) noexcept = default;
  DGonGaussianPoints &operator=(DGonGaussianPoints &&) noexcept = default;
  ~DGonGaussianPoints() noexcept = default;

 protected:  // override virtual methods defined in Base
  void AddFluxDivergence(Column *residual) const override {
    if (Part::kDegrees > 0) {
      for (const Cell &cell : this->part_ptr_->GetLocalCells()) {
        auto i_cell = cell.id();
        auto *data = residual->data() + this->part_ptr_->GetCellDataOffset(i_cell);
        const auto &gauss = cell.gauss();
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          auto const &flux = cell.GetFluxOnGaussianPoint(q);
          auto const &grad = cell.projection_.GetBasisGradientsOnGaussianPoint(q);
          Coeff prod = flux * grad;
          prod *= gauss.GetGlobalWeight(q);
          cell.projection_.AddCoeffTo(prod, data);
        }
      }
    }
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

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_DG_HPP_
