// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_FEM_HPP_
#define MINI_SPATIAL_FEM_HPP_

#include <cassert>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "mini/riemann/concept.hpp"
#include "mini/temporal/ode.hpp"
#include "mini/constant/index.hpp"

namespace mini {
namespace spatial {

using namespace mini::constant::index;

static bool Near(auto const &x, auto const &y) {
  return (x - y).norm() < 1e-12;
}

template <typename P>
class DummySource {
 public:
  using Part = P;
  using Cell = typename Part::Cell;
  using Scalar = typename Cell::Scalar;

  void UpdateCoeff(const Cell &cell, double t_curr, Scalar *coeff) {
  }
};

template <typename Part>
class FiniteElement : public temporal::System<typename Part::Scalar> {
 public:
  using Riemann = typename Part::Riemann;
  using Scalar = typename Part::Scalar;
  using Face = typename Part::Face;
  using Cell = typename Part::Cell;
  using Global = typename Cell::Global;
  using Gauss = typename Cell::Gauss;
  using Projection = typename Cell::Projection;
  using Coeff = typename Projection::Coeff;
  using Value = typename Projection::Value;
  using Temporal = temporal::System<typename Part::Scalar>;
  using Column = typename Temporal::Column;

 protected:
  std::vector<std::string> supersonic_outlet_, solid_wall_;
  using Function = std::function<Value(const Global &, double)>;
  std::unordered_map<std::string, Function> supersonic_inlet_,
      subsonic_inlet_, subsonic_outlet_, smart_boundary_;

  Part *part_ptr_;
  double t_curr_;
  size_t cell_data_size_;

 public:
  explicit FiniteElement(Part *part_ptr)
      : part_ptr_(part_ptr), cell_data_size_(this->part_ptr_->GetCellDataSize()) {
  }
  FiniteElement(const FiniteElement &) = default;
  FiniteElement &operator=(const FiniteElement &) = default;
  FiniteElement(FiniteElement &&) noexcept = default;
  FiniteElement &operator=(FiniteElement &&) noexcept = default;
  ~FiniteElement() noexcept = default;

  virtual const char *name() const {
    return "FEM";
  }

 public:  // set BCs
  template <typename Callable>
  void SetSmartBoundary(const std::string &name, Callable &&func) {
    smart_boundary_[name] = func;
  }
  template <typename Callable>
  void SetSupersonicInlet(const std::string &name, Callable &&func) {
    supersonic_inlet_[name] = func;
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
    for (Cell *cell_ptr: this->part_ptr_->GetLocalCellPointers()) {
      auto i_cell = cell_ptr->id();
      Scalar const *data = column.data() + this->part_ptr_->GetCellDataOffset(i_cell);
      data = cell_ptr->projection().GetCoeffFrom(data);
      assert(data == column.data() + this->part_ptr_->GetCellDataOffset(i_cell + 1));
    }
  }
  Column GetSolutionColumn() const override {
    auto column = Column(cell_data_size_);
    for (const auto &cell : this->part_ptr_->GetLocalCells()) {
      auto i_cell = cell.id();
      Scalar *data = column.data() + this->part_ptr_->GetCellDataOffset(i_cell);
      data = cell.projection().WriteCoeffTo(data);
      assert(data == column.data() + this->part_ptr_->GetCellDataOffset(i_cell + 1));
    }
    return column;
  }
  Column GetResidualColumn() const override {
    this->part_ptr_->ShareGhostCellCoeffs();
    auto residual = Column(cell_data_size_);
    residual.setZero();
    this->AddFluxDivergence(&residual);
    this->AddFluxOnLocalFaces(&residual);
    this->AddFluxOnBoundaries(&residual);
    this->part_ptr_->UpdateGhostCellCoeffs();
    this->AddFluxOnGhostFaces(&residual);
    return residual;
  }

  void AddFluxOnBoundaries(Column *residual) const {
    std::cout << this->name() << "::ApplySolidWall\n";
    std::cout << residual->squaredNorm() << " ";
    this->ApplySolidWall(residual);
    std::cout << residual->squaredNorm() << "\n";
    std::cout << this->name() << "::ApplySupersonicInlet\n";
    std::cout << residual->squaredNorm() << " ";
    this->ApplySupersonicInlet(residual);
    std::cout << residual->squaredNorm() << "\n";
    std::cout << this->name() << "::ApplySupersonicOutlet\n";
    std::cout << residual->squaredNorm() << " ";
    this->ApplySupersonicOutlet(residual);
    std::cout << residual->squaredNorm() << "\n";
    std::cout << this->name() << "::ApplySubsonicInlet\n";
    std::cout << residual->squaredNorm() << " ";
    this->ApplySubsonicInlet(residual);
    std::cout << residual->squaredNorm() << "\n";
    std::cout << this->name() << "::ApplySubsonicOutlet\n";
    std::cout << residual->squaredNorm() << " ";
    this->ApplySubsonicOutlet(residual);
    std::cout << residual->squaredNorm() << "\n";
    std::cout << this->name() << "::ApplySmartBoundary\n";
    std::cout << residual->squaredNorm() << " ";
    this->ApplySmartBoundary(residual);
    std::cout << residual->squaredNorm() << "\n";
  }

 protected:  // declare pure virtual methods to be implemented in subclasses
  virtual void AddFluxDivergence(Column *residual) const = 0;
  virtual void AddFluxOnLocalFaces(Column *residual) const = 0;
  virtual void AddFluxOnGhostFaces(Column *residual) const = 0;
  virtual void ApplySolidWall(Column *residual) const = 0;
  virtual void ApplySupersonicInlet(Column *residual) const = 0;
  virtual void ApplySupersonicOutlet(Column *residual) const = 0;
  virtual void ApplySubsonicInlet(Column *residual) const = 0;
  virtual void ApplySubsonicOutlet(Column *residual) const = 0;
  virtual void ApplySmartBoundary(Column *residual) const = 0;

 protected:
  using FluxMatrix = typename Riemann::FluxMatrix;
  static FluxMatrix GetFluxMatrix(const Projection &projection, int q)
      requires(!mini::riemann::Diffusive<Riemann>) {
    return Riemann::GetFluxMatrix(projection.GetValue(q));
  }
  static FluxMatrix GetFluxMatrix(const Projection &projection, int q)
      requires(mini::riemann::ConvectiveDiffusive<Riemann>) {
    const auto &value = projection.GetValue(q);
    FluxMatrix flux_matrix = Riemann::GetFluxMatrix(value);
    const auto &gradient = projection.GetGlobalGradient(q);
    Riemann::MinusViscousFlux(value, gradient, &flux_matrix);
    return flux_matrix;
  }
};

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_FEM_HPP_
