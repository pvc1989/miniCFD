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
  Column residual_;

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

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_DG_HPP_
