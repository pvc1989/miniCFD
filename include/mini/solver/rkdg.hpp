// Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_SOLVER_RKDG_HPP_
#define MINI_SOLVER_RKDG_HPP_

#include <cassert>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "mini/dataset/part.hpp"

template <typename P>
class DummySource {
 public:
  using Part = P;
  using Cell = typename Part::Cell;
  using Coeff = typename Cell::Projection::Coeff;

  void UpdateCoeff(const Cell &cell, double t_curr, Coeff *coeff) {
  }
};

template <typename P, typename L, typename S>
class RungeKuttaBase {
 public:
  using Part = P;
  using Limiter = L;
  using Source = S;
  using Riemann = typename Part::Riemann;
  using Cell = typename Part::Cell;
  using Face = typename Part::Face;
  using Projection = typename Cell::Projection;
  using Coeff = typename Projection::Coeff;
  using Value = typename Projection::Value;
  using Coord = typename Cell::Coord;

 protected:
  std::vector<Coeff> residual_;
  std::vector<std::string> supersonic_outlet_, solid_wall_;
  using Function = std::function<Value(const Coord&, double)>;
  std::unordered_map<std::string, Function> supersonic_inlet__,
      subsonic_inlet_, subsonic_outlet_;
  Limiter limiter_;
  Source source_;
  double dt_, t_curr_;

 public:
  RungeKuttaBase(double dt, const Limiter &limiter,
      const Source &source = Source())
      : dt_(dt), limiter_(limiter), source_(source) {
    assert(dt > 0.0);
  }
  RungeKuttaBase(const RungeKuttaBase &) = default;
  RungeKuttaBase& operator=(const RungeKuttaBase &) = default;
  RungeKuttaBase(RungeKuttaBase &&) noexcept = default;
  RungeKuttaBase& operator=(RungeKuttaBase &&) noexcept = default;
  ~RungeKuttaBase() noexcept = default;

 public:  // set BCs
  template <typename Callable>
  void SetSupersonicInlet(const std::string &name, Callable&& func) {
    supersonic_inlet__[name] = func;
  }
  template <typename Callable>
  void SetSubsonicInlet(const std::string &name, Callable&& func) {
    subsonic_inlet_[name] = func;
  }
  template <typename Callable>
  void SetSubsonicOutlet(const std::string &name, Callable&& func) {
    subsonic_outlet_[name] = func;
  }
  void SetSolidWall(const std::string &name) {
    solid_wall_.emplace_back(name);
  }
  void SetSupersonicOutlet(const std::string &name) {
    supersonic_outlet_.emplace_back(name);
  }

 public:
  static void ReadFromLocalCells(const Part &part, std::vector<Coeff> *coeffs) {
    coeffs->resize(part.CountLocalCells());
    part.ForEachConstLocalCell([&coeffs](const auto &cell){
      coeffs->at(cell.id())
          = cell.projection_.GetCoeffOnOrthoNormalBasis();
    });
  }
  static void WriteToLocalCells(const std::vector<Coeff> &coeffs, Part *part) {
    assert(coeffs.size() == part->CountLocalCells());
    part->ForEachLocalCell([&coeffs](Cell *cell_ptr){
      Coeff new_coeff = coeffs.at(cell_ptr->id()) * cell_ptr->basis_.coeff();
      cell_ptr->projection_.UpdateCoeffs(new_coeff);
    });
  }
  void InitializeResidual(const Part &part) {
    residual_.resize(part.CountLocalCells());
    for (auto& coeff : residual_) {
      coeff.setZero();
    }
    // Integrate the source term, if there is any.
    if (!std::is_same_v<Source, DummySource<Part>>) {
      part.ForEachConstLocalCell([this](const Cell &cell){
        auto& coeff = this->residual_.at(cell.id());
        source_.UpdateCoeff(cell, this->t_curr_, &coeff);
      });
    }
    // Integrate the dot-product of flux and gradient, if there is any.
    if (Part::kDegrees > 0) {
      part.ForEachConstLocalCell([this](const Cell &cell){
        auto& coeff = this->residual_.at(cell.id());
        const auto& gauss = *(cell.gauss_ptr_);
        auto n = gauss.CountQuadraturePoints();
        for (int q = 0; q < n; ++q) {
          const auto& xyz = gauss.GetGlobalCoord(q);
          Value cv = cell.projection_(xyz);
          auto flux = Riemann::GetFluxMatrix(cv);
          auto grad = cell.basis_.GetGradValue(xyz);
          Coeff prod = flux * grad.transpose();
          prod *= gauss.GetGlobalWeight(q);
          coeff += prod;
        }
      });
    }
  }
  void UpdateLocalResidual(const Part &part) {
    assert(residual_.size() == part.CountLocalCells());
    part.ForEachConstLocalFace([this](const Face &face){
      const auto& gauss = *(face.gauss_ptr_);
      const auto& holder = *(face.holder_);
      const auto& sharer = *(face.sharer_);
      const auto& riemann = (face.riemann_);
      auto n = gauss.CountQuadraturePoints();
      for (int q = 0; q < n; ++q) {
        const auto& coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.projection_(coord);
        Value u_sharer = sharer.projection_(coord);
        Value flux = riemann.GetFluxOnTimeAxis(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        this->residual_.at(holder.id())
            -= flux * holder.basis_(coord).transpose();
        this->residual_.at(sharer.id())
            += flux * sharer.basis_(coord).transpose();
      }
    });
  }
  void UpdateGhostResidual(const Part &part) {
    assert(residual_.size() == part.CountLocalCells());
    part.ForEachConstGhostFace([this](const Face &face){
      const auto& gauss = *(face.gauss_ptr_);
      const auto& holder = *(face.holder_);
      const auto& sharer = *(face.sharer_);
      const auto& riemann = (face.riemann_);
      auto n = gauss.CountQuadraturePoints();
      for (int q = 0; q < n; ++q) {
        const auto& coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.projection_(coord);
        Value u_sharer = sharer.projection_(coord);
        Value flux = riemann.GetFluxOnTimeAxis(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        Coeff temp = flux * holder.basis_(coord).transpose();
        this->residual_.at(holder.id()) -= temp;
      }
    });
  }
  void ApplySolidWall(const Part &part) {
    auto visit = [this](const Face &face){
      const auto& gauss = *(face.gauss_ptr_);
      assert(face.sharer_ == nullptr);
      const auto& holder = *(face.holder_);
      const auto& riemann = (face.riemann_);
      auto n = gauss.CountQuadraturePoints();
      for (int q = 0; q < n; ++q) {
        const auto& coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.projection_(coord);
        Value flux = riemann.GetFluxOnSolidWall(u_holder);
        flux *= gauss.GetGlobalWeight(q);
        this->residual_.at(holder.id())
            -= flux * holder.basis_(coord).transpose();
      }
    };
    for (const auto& name : solid_wall_) {
      part.ForEachConstBoundaryFace(visit, name);
    }
  }
  void ApplySupersonicOutlet(const Part &part) {
    auto visit = [this](const Face &face){
      const auto& gauss = *(face.gauss_ptr_);
      assert(face.sharer_ == nullptr);
      const auto& holder = *(face.holder_);
      const auto& riemann = (face.riemann_);
      auto n = gauss.CountQuadraturePoints();
      for (int q = 0; q < n; ++q) {
        const auto& coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.projection_(coord);
        Value flux = riemann.GetFluxOnSupersonicOutlet(u_holder);
        flux *= gauss.GetGlobalWeight(q);
        this->residual_.at(holder.id())
            -= flux * holder.basis_(coord).transpose();
      }
    };
    for (const auto& name : supersonic_outlet_) {
      part.ForEachConstBoundaryFace(visit, name);
    }
  }
  void ApplySupersonicInlet(const Part &part) {
    for (auto iter = supersonic_inlet__.begin();
        iter != supersonic_inlet__.end(); ++iter) {
      auto visit = [this, iter](const Face &face){
        const auto& gauss = *(face.gauss_ptr_);
        assert(face.sharer_ == nullptr);
        const auto& holder = *(face.holder_);
        const auto& riemann = (face.riemann_);
        auto n = gauss.CountQuadraturePoints();
        for (int q = 0; q < n; ++q) {
          const auto& coord = gauss.GetGlobalCoord(q);
          Value u_given = iter->second(coord, this->t_curr_);
          Value flux = riemann.GetFluxOnSupersonicInlet(u_given);
          flux *= gauss.GetGlobalWeight(q);
          this->residual_.at(holder.id())
              -= flux * holder.basis_(coord).transpose();
        }
      };
      part.ForEachConstBoundaryFace(visit, iter->first);
    }
  }
  void ApplySubsonicInlet(const Part &part) {
    for (auto iter = subsonic_inlet_.begin(); iter != subsonic_inlet_.end();
        ++iter) {
      auto visit = [this, iter](const Face &face){
        const auto& gauss = *(face.gauss_ptr_);
        assert(face.sharer_ == nullptr);
        const auto& holder = *(face.holder_);
        const auto& riemann = (face.riemann_);
        auto n = gauss.CountQuadraturePoints();
        for (int q = 0; q < n; ++q) {
          const auto& coord = gauss.GetGlobalCoord(q);
          Value u_inner = holder.projection_(coord);
          Value u_given = iter->second(coord, this->t_curr_);
          Value flux = riemann.GetFluxOnSubsonicInlet(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(q);
          this->residual_.at(holder.id())
              -= flux * holder.basis_(coord).transpose();
        }
      };
      part.ForEachConstBoundaryFace(visit, iter->first);
    }
  }
  void ApplySubsonicOutlet(const Part &part) {
    for (auto iter = subsonic_outlet_.begin(); iter != subsonic_outlet_.end();
        ++iter) {
      auto visit = [this, iter](const Face &face){
        const auto& gauss = *(face.gauss_ptr_);
        assert(face.sharer_ == nullptr);
        const auto& holder = *(face.holder_);
        const auto& riemann = (face.riemann_);
        auto n = gauss.CountQuadraturePoints();
        for (int q = 0; q < n; ++q) {
          const auto& coord = gauss.GetGlobalCoord(q);
          Value u_inner = holder.projection_(coord);
          Value u_given = iter->second(coord, this->t_curr_);
          Value flux = riemann.GetFluxOnSubsonicOutlet(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(q);
          this->residual_.at(holder.id())
              -= flux * holder.basis_(coord).transpose();
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
  }
};

template <int kOrders, typename Part, typename Limiter,
    typename Source = DummySource<Part>
> struct RungeKutta;

template <typename P, typename L, typename S>
struct RungeKutta<1, P, L, S>
    : public RungeKuttaBase<P, L, S> {
 private:
  using Base = RungeKuttaBase<P, L, S>;

 public:
  using Part = typename Base::Part;
  using Riemann = typename Base::Riemann;
  using Limiter = typename Base::Limiter;
  using Source = typename Base::Source;
  using Cell = typename Base::Cell;
  using Face = typename Base::Face;
  using Projection = typename Base::Projection;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;

 private:
  std::vector<Coeff> u_old_, u_new_;

 public:
  using Base::Base;
  RungeKutta(const RungeKutta &) = default;
  RungeKutta& operator=(const RungeKutta &) = default;
  RungeKutta(RungeKutta &&) noexcept = default;
  RungeKutta& operator=(RungeKutta &&) noexcept = default;
  ~RungeKutta() noexcept = default;

 public:
  void Update(Part *part_ptr, double t_curr) {
    const Part &part = *part_ptr;
    this->t_curr_ = t_curr;

    Base::ReadFromLocalCells(part, &u_old_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeResidual(part);
    this->UpdateLocalResidual(part);
    this->UpdateBoundaryResidual(part);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostResidual(part);
    this->SolveFrac11();
    Base::WriteToLocalCells(u_new_, part_ptr);
    part_ptr->Reconstruct(this->limiter_);
  }

 private:
  void SolveFrac11() {
    auto n_cells = u_old_.size();
    assert(n_cells == this->residual_.size());
    u_new_ = this->residual_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_new_[i_cell] *= this->dt_;
      u_new_[i_cell] += u_old_[i_cell];
    }
  }
};

template <typename P, typename L, typename S>
struct RungeKutta<2, P, L, S>
    : public RungeKuttaBase<P, L, S> {
 private:
  using Base = RungeKuttaBase<P, L, S>;

 public:
  using Part = typename Base::Part;
  using Riemann = typename Base::Riemann;
  using Limiter = typename Base::Limiter;
  using Source = typename Base::Source;
  using Cell = typename Base::Cell;
  using Face = typename Base::Face;
  using Projection = typename Base::Projection;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;

 private:
  std::vector<Coeff> u_old_, u_frac12_, u_new_;

 public:
  using Base::Base;
  RungeKutta(const RungeKutta &) = default;
  RungeKutta& operator=(const RungeKutta &) = default;
  RungeKutta(RungeKutta &&) noexcept = default;
  RungeKutta& operator=(RungeKutta &&) noexcept = default;
  ~RungeKutta() noexcept = default;

 public:
  void Update(Part *part_ptr, double t_curr) {
    const Part &part = *part_ptr;
    this->t_curr_ = t_curr;

    Base::ReadFromLocalCells(part, &u_old_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeResidual(part);
    this->UpdateLocalResidual(part);
    this->UpdateBoundaryResidual(part);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostResidual(part);
    this->SolveFrac12();
    Base::WriteToLocalCells(u_frac12_, part_ptr);
    part_ptr->Reconstruct(this->limiter_);

    Base::ReadFromLocalCells(part, &u_frac12_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeResidual(part);
    this->UpdateLocalResidual(part);
    this->UpdateBoundaryResidual(part);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostResidual(part);
    this->SolveFrac22();
    Base::WriteToLocalCells(u_new_, part_ptr);
    part_ptr->Reconstruct(this->limiter_);
  }

 private:
  void SolveFrac12() {
    auto n_cells = u_old_.size();
    assert(n_cells == this->residual_.size());
    u_frac12_ = this->residual_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_frac12_[i_cell] *= this->dt_;
      u_frac12_[i_cell] += u_old_[i_cell];
    }
  }
  void SolveFrac22() {
    auto n_cells = u_old_.size();
    assert(n_cells == this->residual_.size());
    assert(n_cells == u_frac12_.size());
    u_new_ = this->residual_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_new_[i_cell] *= this->dt_;
      u_new_[i_cell] += u_frac12_[i_cell];
      u_new_[i_cell] += u_old_[i_cell];
      u_new_[i_cell] /= 2;
    }
  }
};

template <typename P, typename L, typename S>
struct RungeKutta<3, P, L, S>
    : public RungeKuttaBase<P, L, S> {
 private:
  using Base = RungeKuttaBase<P, L, S>;

 public:
  using Part = typename Base::Part;
  using Riemann = typename Base::Riemann;
  using Limiter = typename Base::Limiter;
  using Source = typename Base::Source;
  using Cell = typename Base::Cell;
  using Face = typename Base::Face;
  using Projection = typename Base::Projection;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;

 private:
  std::vector<Coeff> u_old_, u_frac13_, u_frac23_, u_new_;

 public:
  using Base::Base;
  RungeKutta(const RungeKutta &) = default;
  RungeKutta& operator=(const RungeKutta &) = default;
  RungeKutta(RungeKutta &&) noexcept = default;
  RungeKutta& operator=(RungeKutta &&) noexcept = default;
  ~RungeKutta() noexcept = default;

 public:
  void Update(Part *part_ptr, double t_curr) {
    const Part &part = *part_ptr;
    this->t_curr_ = t_curr;

    Base::ReadFromLocalCells(part, &u_old_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeResidual(part);
    this->UpdateLocalResidual(part);
    this->UpdateBoundaryResidual(part);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostResidual(part);
    this->SolveFrac13();
    Base::WriteToLocalCells(u_frac13_, part_ptr);
    part_ptr->Reconstruct(this->limiter_);

    Base::ReadFromLocalCells(part, &u_frac13_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeResidual(part);
    this->UpdateLocalResidual(part);
    this->UpdateBoundaryResidual(part);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostResidual(part);
    this->SolveFrac23();
    Base::WriteToLocalCells(u_frac23_, part_ptr);
    part_ptr->Reconstruct(this->limiter_);

    Base::ReadFromLocalCells(part, &u_frac23_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeResidual(part);
    this->UpdateLocalResidual(part);
    this->UpdateBoundaryResidual(part);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostResidual(part);
    this->SolveFrac33();
    Base::WriteToLocalCells(u_new_, part_ptr);
    part_ptr->Reconstruct(this->limiter_);
  }

 private:
  void SolveFrac13() {
    auto n_cells = u_old_.size();
    assert(n_cells == this->residual_.size());
    u_frac13_ = this->residual_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_frac13_[i_cell] *= this->dt_;
      u_frac13_[i_cell] += u_old_[i_cell];
    }
  }
  void SolveFrac23() {
    auto n_cells = u_old_.size();
    assert(n_cells == this->residual_.size());
    assert(n_cells == u_frac13_.size());
    u_frac23_ = this->residual_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_frac23_[i_cell] *= this->dt_;
      u_frac23_[i_cell] += u_frac13_[i_cell];
      u_frac23_[i_cell] += u_old_[i_cell];
      u_frac23_[i_cell] += u_old_[i_cell];
      u_frac23_[i_cell] += u_old_[i_cell];
      u_frac23_[i_cell] /= 4;
    }
  }
  void SolveFrac33() {
    auto n_cells = u_old_.size();
    assert(n_cells == this->residual_.size());
    assert(n_cells == u_frac13_.size());
    assert(n_cells == u_frac23_.size());
    u_new_ = this->residual_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_new_[i_cell] *= 2 * this->dt_;
      u_new_[i_cell] += u_frac23_[i_cell];
      u_new_[i_cell] += u_frac23_[i_cell];
      u_new_[i_cell] += u_old_[i_cell];
      u_new_[i_cell] /= 3;
    }
  }
};

#endif  // MINI_SOLVER_RKDG_HPP_
