#include <cassert>
#include <vector>
#include <stdexcept>
#include "mini/mesh/part.hpp"

template <typename PartType, typename RiemannType>
class RungeKuttaBase {
 public:
  using Riemann = RiemannType;
  using Part = PartType;
  using Cell = typename Part::CellType;
  using Face = typename Part::FaceType;
  using Projection = typename Cell::Projection;
  using Coeff = typename Projection::Coeff;
  using Value = typename Projection::Value;

 protected:
  std::vector<Coeff> rhs_;
  std::vector<std::vector<Riemann>> riemann_solvers_;
  double dt_;

 public:
  explicit RungeKuttaBase(double dt)
      : dt_(dt) {
    assert(dt > 0.0);
  }
  RungeKuttaBase(const RungeKuttaBase &) = default;
  RungeKuttaBase& operator=(const RungeKuttaBase &) = default;
  RungeKuttaBase(RungeKuttaBase &&) noexcept = default;
  RungeKuttaBase& operator=(RungeKuttaBase &&) noexcept = default;
  ~RungeKuttaBase() noexcept = default;

 public:
  static void ReadFromLocalCells(const Part &part, std::vector<Coeff> *coeffs) {
    coeffs->resize(part.CountLocalCells());
    part.ForEachLocalCell([&coeffs](const auto &cell){
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
  void BuildRiemannSolvers(const Part &part) {
    auto riemann_builder = [this](const Face &face){
      auto& solvers = this->riemann_solvers_.emplace_back();
      assert(face.id() == this->riemann_solvers_.size());
      auto& gauss = face.gauss();
      solvers.resize(gauss.CountQuadPoints());
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        auto& frame = gauss.GetNormalFrame(q);
        solvers[q].Rotate(frame);
      }
    };
    part.ForEachLocalFace(riemann_builder);
    part.ForEachGhostFace(riemann_builder);
    part.ForEachSolidFace(riemann_builder);
  }
  void InitializeRhs(const Part &part) {
    rhs_.resize(part.CountLocalCells());
    for (auto& coeff : rhs_) {
      coeff.setZero();
    }
    part.ForEachLocalCell([this](const Cell &cell){
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
    part.ForEachLocalFace([this](const Face &face){
      const auto& gauss = *(face.gauss_ptr_);
      const auto& holder = *(face.holder_);
      const auto& sharer = *(face.sharer_);
      auto& riemann_solvers = riemann_solvers_.at(face.id());
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        const auto& coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.projection_(coord);
        Value u_sharer = sharer.projection_(coord);
        auto& riemann = riemann_solvers[q];
        Value flux = riemann.GetFluxOnTimeAxis(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        this->rhs_.at(holder.id()) -= flux * holder.basis_(coord).transpose();
        this->rhs_.at(sharer.id()) += flux * sharer.basis_(coord).transpose();
      }
    });
  }
  void UpdateGhostRhs(const Part &part) {
    assert(rhs_.size() == part.CountLocalCells());
    part.ForEachGhostFace([this](const Face &face){
      const auto& gauss = *(face.gauss_ptr_);
      const auto& holder = *(face.holder_);
      const auto& sharer = *(face.sharer_);
      auto& riemann_solvers = riemann_solvers_.at(face.id());
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        const auto& coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.projection_(coord);
        Value u_sharer = sharer.projection_(coord);
        auto& riemann = riemann_solvers[q];
        Value flux = riemann.GetFluxOnTimeAxis(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        Coeff temp = flux * holder.basis_(coord).transpose();
        this->rhs_.at(holder.id()) -= temp;
      }
    });
  }
  void UpdateBoundaryRhs(const Part &part) {
    part.ForEachSolidFace([this](const Face &face){
      const auto& gauss = *(face.gauss_ptr_);
      assert(face.sharer_ == nullptr);
      const auto& holder = *(face.holder_);
      auto& riemann_solvers = riemann_solvers_.at(face.id());
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        const auto& coord = gauss.GetGlobalCoord(q);
        auto& riemann = riemann_solvers[q];
        Value u_holder = holder.projection_(coord);
        Value flux = riemann.GetFluxOnSolidWall(u_holder);
        flux *= gauss.GetGlobalWeight(q);
        this->rhs_.at(holder.id()) -= flux * holder.basis_(coord).transpose();
      }
    });
  }
};

template <int kTemporalAccuracy, typename PartType, typename RiemannType>
struct RungeKutta;

template <typename PartType, typename RiemannType>
struct RungeKutta<1, PartType, RiemannType>
    : public RungeKuttaBase<PartType, RiemannType> {
 private:
  using Base = RungeKuttaBase<PartType, RiemannType>;

 public:
  using Part = typename Base::Part;
  using Riemann = typename Base::Riemann;
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
  template <class Limiter>
  void Update(Part *part_ptr, Limiter& limiter) {
    const Part &part = *part_ptr;  // TODO: change to const ref

    Base::ReadFromLocalCells(part, &u_old_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeRhs(part);
    this->UpdateLocalRhs(part);
    this->UpdateBoundaryRhs(part);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostRhs(part);
    auto n_cells = u_old_.size();
    assert(n_cells == rhs_.size());
    u_new_ = this->rhs_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_new_[i_cell] *= this->dt_;
      u_new_[i_cell] += u_old_[i_cell];
    }
    Base::WriteToLocalCells(u_new_, part_ptr);
    // part_ptr->Reconstruct(limiter);  // only for high order methods
  }
};

template <typename PartType, typename RiemannType>
struct RungeKutta<3, PartType, RiemannType>
    : public RungeKuttaBase<PartType, RiemannType> {
 private:
  using Base = RungeKuttaBase<PartType, RiemannType>;

 public:
  using Part = typename Base::Part;
  using Riemann = typename Base::Riemann;
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
  template <class Limiter>
  void Update(Part *part_ptr, Limiter& limiter) {
    const Part &part = *part_ptr;  // TODO: change to const ref

    Base::ReadFromLocalCells(part, &u_old_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeRhs(part);
    this->UpdateLocalRhs(part);
    this->UpdateBoundaryRhs(part);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostRhs(part);
    this->SolveFrac13();
    Base::WriteToLocalCells(u_frac13_, part_ptr);
    part_ptr->Reconstruct(limiter);

    Base::ReadFromLocalCells(part, &u_frac13_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeRhs(part);
    this->UpdateLocalRhs(part);
    this->UpdateBoundaryRhs(part);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostRhs(part);
    this->SolveFrac23();
    Base::WriteToLocalCells(u_frac23_, part_ptr);
    part_ptr->Reconstruct(limiter);

    Base::ReadFromLocalCells(part, &u_frac23_);
    part_ptr->ShareGhostCellCoeffs();
    this->InitializeRhs(part);
    this->UpdateLocalRhs(part);
    this->UpdateBoundaryRhs(part);
    part_ptr->UpdateGhostCellCoeffs();
    this->UpdateGhostRhs(part);
    this->SolveFrac33();
    Base::WriteToLocalCells(u_new_, part_ptr);
    part_ptr->Reconstruct(limiter);
  }

 private:
  void SolveFrac13() {
    auto n_cells = u_old_.size();
    assert(n_cells == rhs_.size());
    u_frac13_ = this->rhs_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_frac13_[i_cell] *= this->dt_;
      u_frac13_[i_cell] += u_old_[i_cell];
    }
  }
  void SolveFrac23() {
    auto n_cells = u_old_.size();
    assert(n_cells == rhs_.size());
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
    assert(n_cells == rhs_.size());
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
