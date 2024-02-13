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

#include "mini/algebra/eigen.hpp"
#include "mini/riemann/concept.hpp"
#include "mini/temporal/ode.hpp"
#include "mini/spatial/fem.hpp"


namespace mini {
namespace spatial {

template <typename Part>
class EnergyBasedViscosity : public FiniteElement<Part> {
 public:
  using Base = FiniteElement<Part>;
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
  using CellToFlux = typename Base::CellToFlux;
  using DampingMatrix = algebra::Matrix<Scalar, Cell::N, Cell::N>;

 private:
  Base *base_ptr_;

 public:
  explicit EnergyBasedViscosity(Base *base_ptr)
      : Base(base_ptr->part_ptr()), base_ptr_(base_ptr) {
  }
  EnergyBasedViscosity(const EnergyBasedViscosity &) = default;
  EnergyBasedViscosity &operator=(const EnergyBasedViscosity &) = default;
  EnergyBasedViscosity(EnergyBasedViscosity &&) noexcept = default;
  EnergyBasedViscosity &operator=(EnergyBasedViscosity &&) noexcept = default;
  ~EnergyBasedViscosity() noexcept = default;

  Base const &base() const {
    return *base_ptr_;
  }
  Part const &part() const {
    return base().part();
  }

  std::string name() const override {
    return base().name() + "EnergyBasedViscosity";
  }

 protected:  // data for generating artificial viscosity
  std::vector<DampingMatrix> damping_matrices_;

 public:  // methods for generating artificial viscosity
  std::vector<DampingMatrix> BuildDampingMatrices() const {
    auto matrices = std::vector<DampingMatrix>(part().CountLocalCells());
    return matrices;
  }

 public:  // override virtual methods defined in Base
  Column GetResidualColumn() const override {
    return base().GetResidualColumn();
  }
  Column GetSolutionColumn() const override {
    return base().GetSolutionColumn();
  }
  void SetSolutionColumn(Column const &column) override {
    base_ptr_->SetSolutionColumn(column);
  }
  void SetTime(double t_curr) override {
    base_ptr_->SetTime(t_curr);
  }

 protected:  // override virtual methods defined in Base
  void AddFluxDivergence(CellToFlux cell_to_flux, Cell const &cell,
      Scalar *data) const override {
    base().AddFluxDivergence(cell_to_flux, cell, data);
  }
  void AddFluxDivergence(CellToFlux cell_to_flux,
      Column *residual) const override {
    base().AddFluxDivergence(cell_to_flux, residual);
  }
  void AddFluxOnGhostFaces(Column *residual) const override {
    base().AddFluxOnGhostFaces(residual);
  }
  void AddFluxOnLocalFaces(Column *residual) const override {
    base().AddFluxOnLocalFaces(residual);
  }
  void ApplySmartBoundary(Column *residual) const override {
    base().ApplySmartBoundary(residual);
  }
  void ApplySolidWall(Column *residual) const override {
    base().ApplySolidWall(residual);
  }
  void ApplySubsonicInlet(Column *residual) const override {
    base().ApplySubsonicInlet(residual);
  }
  void ApplySubsonicOutlet(Column *residual) const override {
    base().ApplySubsonicOutlet(residual);
  }
  void ApplySupersonicInlet(Column *residual) const override {
    base().ApplySupersonicInlet(residual);
  }
  void ApplySupersonicOutlet(Column *residual) const override {
    base().ApplySupersonicOutlet(residual);
  }
};

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_VISCOSITY_HPP_
