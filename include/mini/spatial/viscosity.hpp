// Copyright 2024 PEI Weicheng
#ifndef MINI_SPATIAL_VISCOSITY_HPP_
#define MINI_SPATIAL_VISCOSITY_HPP_

#include <concepts>

#include <cassert>
#include <fstream>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "mini/riemann/concept.hpp"
#include "mini/temporal/ode.hpp"
#include "mini/spatial/fem.hpp"


namespace mini {
namespace spatial {

template <typename Base>
    requires std::derived_from<Base, FiniteElement<typename Base::Part>>
class EnergyBasedViscosity : public Base {
 public:
  using Part = typename Base::Part;
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
  explicit EnergyBasedViscosity(Part *part_ptr)
      : Base(part_ptr) {
  }
  EnergyBasedViscosity(const EnergyBasedViscosity &) = default;
  EnergyBasedViscosity &operator=(const EnergyBasedViscosity &) = default;
  EnergyBasedViscosity(EnergyBasedViscosity &&) noexcept = default;
  EnergyBasedViscosity &operator=(EnergyBasedViscosity &&) noexcept = default;
  ~EnergyBasedViscosity() noexcept = default;

  std::string name() const override {
    return this->Base::name() + "EnergyBasedViscosity";
  }

 public:  // override virtual methods defined in Base
  Column GetResidualColumn() const override {
    auto residual = this->Base::GetResidualColumn();
    return residual;
  }

 protected:  // override virtual methods defined in Base
  void AddFluxDivergence(Column *residual) const override {
    this->Base::AddFluxDivergence(residual);
  }
  void AddFluxOnLocalFaces(Column *residual) const override {
    this->Base::AddFluxOnLocalFaces(residual);
  }
  void AddFluxOnGhostFaces(Column *residual) const override {
    this->Base::AddFluxOnLocalFaces(residual);
  }
};

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_VISCOSITY_HPP_
