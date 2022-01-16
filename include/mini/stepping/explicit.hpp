// Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_STEPPING_EXPLICIT_HPP_
#define MINI_STEPPING_EXPLICIT_HPP_

#include <cassert>
#include <functional>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "mini/mesh/part.hpp"

template <typename P, typename L>
class RungeKuttaBase {
 public:
  using Part = P;
  using Limiter = L;
  using Riemann = typename Part::Riemann;
  using Cell = typename Part::Cell;
  using Face = typename Part::Face;
  using Projection = typename Cell::Projection;
  using Coeff = typename Projection::Coeff;
  using Value = typename Projection::Value;
  using Coord = typename Cell::Coord;

 protected:
  std::vector<Coeff> rhs_;
  std::vector<std::string> free_bc_, solid_bc_;
  using Function = std::function<Value(const Coord&, double)>;
  std::unordered_map<std::string, Function> prescribed_bc_;
  Limiter limiter_;
  double dt_;

 public:
  RungeKuttaBase(double dt, const Limiter &limiter)
      : dt_(dt), limiter_(limiter) {
    assert(dt > 0.0);
  }
  RungeKuttaBase(const RungeKuttaBase &) = default;
  RungeKuttaBase& operator=(const RungeKuttaBase &) = default;
  RungeKuttaBase(RungeKuttaBase &&) noexcept = default;
  RungeKuttaBase& operator=(RungeKuttaBase &&) noexcept = default;
  ~RungeKuttaBase() noexcept = default;

 public:  // set BCs
  template <typename Callable>
  void SetPrescribedBC(const std::string &name, Callable&& func) {
    prescribed_bc_[name] = func;
  }
  void SetSolidWallBC(const std::string &name) {
    solid_bc_.emplace_back(name);
  }
  void SetFreeOutletBC(const std::string &name) {
    free_bc_.emplace_back(name);
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
  void InitializeRhs(const Part &part) {
    rhs_.resize(part.CountLocalCells());
    for (auto& coeff : rhs_) {
      coeff.setZero();
    }
    part.ForEachConstLocalCell([this](const Cell &cell){
      auto& r = this->rhs_.at(cell.id());
      const auto& gauss = *(cell.gauss_ptr_);
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        const auto& xyz = gauss.GetGlobalCoord(q);
        Value cv = cell.projection_(xyz);
        auto flux = Riemann::GetFluxMatrix(cv);
        auto grad = cell.basis_.GetGradValue(xyz);
        Coeff prod = flux * grad.transpose();
        prod *= gauss.GetGlobalWeight(q);
        r += prod;
      }
    });
  }
  void UpdateLocalRhs(const Part &part) {
    assert(rhs_.size() == part.CountLocalCells());
    part.ForEachConstLocalFace([this](const Face &face){
      const auto& gauss = *(face.gauss_ptr_);
      const auto& holder = *(face.holder_);
      const auto& sharer = *(face.sharer_);
      auto& riemann = const_cast<Riemann &>(face.riemann_);
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        const auto& coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.projection_(coord);
        Value u_sharer = sharer.projection_(coord);
        Value flux = riemann.GetFluxOnTimeAxis(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        this->rhs_.at(holder.id()) -= flux * holder.basis_(coord).transpose();
        this->rhs_.at(sharer.id()) += flux * sharer.basis_(coord).transpose();
      }
    });
  }
  void UpdateGhostRhs(const Part &part) {
    assert(rhs_.size() == part.CountLocalCells());
    part.ForEachConstGhostFace([this](const Face &face){
      const auto& gauss = *(face.gauss_ptr_);
      const auto& holder = *(face.holder_);
      const auto& sharer = *(face.sharer_);
      auto& riemann = const_cast<Riemann &>(face.riemann_);
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        const auto& coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.projection_(coord);
        Value u_sharer = sharer.projection_(coord);
        Value flux = riemann.GetFluxOnTimeAxis(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        Coeff temp = flux * holder.basis_(coord).transpose();
        this->rhs_.at(holder.id()) -= temp;
      }
    });
  }
  void ApplySolidWallBC(const Part &part) {
    auto visit = [this](const Face &face){
      const auto& gauss = *(face.gauss_ptr_);
      assert(face.sharer_ == nullptr);
      const auto& holder = *(face.holder_);
      auto& riemann = const_cast<Riemann &>(face.riemann_);
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        const auto& coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.projection_(coord);
        Value flux = riemann.GetFluxOnSolidWall(u_holder);
        flux *= gauss.GetGlobalWeight(q);
        this->rhs_.at(holder.id()) -= flux * holder.basis_(coord).transpose();
      }
    };
    for (const auto& name : solid_bc_) {
      part.ForEachConstBoundaryFace(visit, name);
    }
  }
  void ApplyFreeOutletBC(const Part &part) {
    auto visit = [this](const Face &face){
      const auto& gauss = *(face.gauss_ptr_);
      assert(face.sharer_ == nullptr);
      const auto& holder = *(face.holder_);
      auto& riemann = const_cast<Riemann &>(face.riemann_);
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        const auto& coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.projection_(coord);
        Value flux = riemann.GetFluxOnFreeWall(u_holder);
        flux *= gauss.GetGlobalWeight(q);
        this->rhs_.at(holder.id()) -= flux * holder.basis_(coord).transpose();
      }
    };
    for (const auto& name : free_bc_) {
      part.ForEachConstBoundaryFace(visit, name);
    }
  }
  void ApplyPrescribedBC(const Part &part, double t_curr) {
    for (auto iter = prescribed_bc_.begin(); iter != prescribed_bc_.end();
        ++iter) {
      auto visit = [this, t_curr, iter](const Face &face){
        const auto& gauss = *(face.gauss_ptr_);
        assert(face.sharer_ == nullptr);
        const auto& holder = *(face.holder_);
        auto& riemann = const_cast<Riemann &>(face.riemann_);
        for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
          const auto& coord = gauss.GetGlobalCoord(q);
          Value u_given = iter->second(coord, t_curr);
          Value flux = riemann.GetRotatedFlux(u_given);
          flux *= gauss.GetGlobalWeight(q);
          this->rhs_.at(holder.id()) -= flux * holder.basis_(coord).transpose();
        }
      };
      part.ForEachConstBoundaryFace(visit, iter->first);
    }
  }
  void UpdateBoundaryRhs(const Part &part, double t_curr) {
    ApplySolidWallBC(part);
    ApplyFreeOutletBC(part);
    ApplyPrescribedBC(part, t_curr);
  }
};

template <int kSteps, typename Part, typename Limiter>
struct RungeKutta;

template <typename P, typename L>
struct RungeKutta<1, P, L>
    : public RungeKuttaBase<P, L> {
 private:
  using Base = RungeKuttaBase<P, L>;

 public:
  using Part = typename Base::Part;
  using Riemann = typename Base::Riemann;
  using Limiter = typename Base::Limiter;
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

    Base::ReadFromLocalCells(part, &u_old_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeRhs(part);
    this->UpdateLocalRhs(part);
    this->UpdateBoundaryRhs(part, t_curr);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostRhs(part);
    auto n_cells = u_old_.size();
    assert(n_cells == this->rhs_.size());
    u_new_ = this->rhs_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_new_[i_cell] *= this->dt_;
      u_new_[i_cell] += u_old_[i_cell];
    }
    Base::WriteToLocalCells(u_new_, part_ptr);
    // part_ptr->Reconstruct(this->limiter_);  // only for high order methods
  }
};

template <typename P, typename L>
struct RungeKutta<3, P, L>
    : public RungeKuttaBase<P, L> {
 private:
  using Base = RungeKuttaBase<P, L>;

 public:
  using Part = typename Base::Part;
  using Riemann = typename Base::Riemann;
  using Limiter = typename Base::Limiter;
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

    Base::ReadFromLocalCells(part, &u_old_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeRhs(part);
    this->UpdateLocalRhs(part);
    this->UpdateBoundaryRhs(part, t_curr);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostRhs(part);
    this->SolveFrac13();
    Base::WriteToLocalCells(u_frac13_, part_ptr);
    part_ptr->Reconstruct(this->limiter_);

    Base::ReadFromLocalCells(part, &u_frac13_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeRhs(part);
    this->UpdateLocalRhs(part);
    this->UpdateBoundaryRhs(part, t_curr);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostRhs(part);
    this->SolveFrac23();
    Base::WriteToLocalCells(u_frac23_, part_ptr);
    part_ptr->Reconstruct(this->limiter_);

    Base::ReadFromLocalCells(part, &u_frac23_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeRhs(part);
    this->UpdateLocalRhs(part);
    this->UpdateBoundaryRhs(part, t_curr);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostRhs(part);
    this->SolveFrac33();
    Base::WriteToLocalCells(u_new_, part_ptr);
    part_ptr->Reconstruct(this->limiter_);
  }

 private:
  void SolveFrac13() {
    auto n_cells = u_old_.size();
    assert(n_cells == this->rhs_.size());
    u_frac13_ = this->rhs_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_frac13_[i_cell] *= this->dt_;
      u_frac13_[i_cell] += u_old_[i_cell];
    }
  }
  void SolveFrac23() {
    auto n_cells = u_old_.size();
    assert(n_cells == this->rhs_.size());
    assert(n_cells == u_frac13_.size());
    u_frac23_ = this->rhs_;
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
    assert(n_cells == this->rhs_.size());
    assert(n_cells == u_frac13_.size());
    assert(n_cells == u_frac23_.size());
    u_new_ = this->rhs_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_new_[i_cell] *= 2 * this->dt_;
      u_new_[i_cell] += u_frac23_[i_cell];
      u_new_[i_cell] += u_frac23_[i_cell];
      u_new_[i_cell] += u_old_[i_cell];
      u_new_[i_cell] /= 3;
    }
  }
};

#endif  // MINI_STEPPING_EXPLICIT_HPP_
