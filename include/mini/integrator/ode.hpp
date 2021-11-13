#include <cassert>
#include <vector>
#include "mini/mesh/part.hpp"

template <typename Part, int kOrder>
struct RungeKutta;

template <typename Part>
struct RungeKutta<Part, 3/* kOrder */> {
  using MyPart = Part;
  using MyCell = typename MyPart::CellType;
  using MyFace = typename MyPart::FaceType;
  using Projection = typename MyCell::Projection;
  using Coeff = typename Projection::Coeff;
  using Value = typename Projection::Value;
  using Gas = typename MyFace::Gas;

  std::vector<Coeff> u_old_, u_frac13_, u_frac23_, u_new_;
  std::vector<Coeff> rhs_;
  double dt_;

  explicit RungeKutta(double dt)
      : dt_(dt) {
    assert(dt > 0.0);
  }
  RungeKutta(const RungeKutta &) = default;
  RungeKutta& operator=(const RungeKutta &) = default;
  RungeKutta(RungeKutta &&) noexcept = default;
  RungeKutta& operator=(RungeKutta &&) noexcept = default;
  ~RungeKutta() noexcept = default;

  static void ReadFromLocalCells(const MyPart &part, std::vector<Coeff> *coeffs) {
    coeffs->resize(part.CountLocalCells());
    part.ForEachLocalCell([&coeffs](const auto &cell){
      coeffs->at(cell.id()) = cell.projection_.coeff();
    });
  }
  static void WriteToLocalCells(const std::vector<Coeff> &coeffs, MyPart *part) {
    assert(coeffs.size() == part->CountLocalCells());
    part->ForEachLocalCell([&coeffs](auto &cell){
      cell.projection_.UpdateCoeffs(coeffs.at(cell.id()));
    });
  }
  static void UpdateLocalRhs(MyPart &part, std::vector<Coeff> *rhs) {
    assert(rhs->size() == part.CountLocalCells());
    part.ForEachLocalFace([&rhs](MyFace &face){
      auto& gauss = *(face.gauss_ptr_);
      std::cout << gauss.area() << std::endl;
      auto& holder = *(face.holder_);
      auto& sharer = *(face.sharer_);
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        auto& coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.projection_(coord);
        Value u_sharer = sharer.projection_(coord);
        auto& riemann = face.GetRiemann(q);
        Value flux = riemann.GetFluxOnTimeAxis(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        rhs->at(holder.id()) -= flux * holder.basis_(coord).transpose();
        rhs->at(sharer.id()) += flux * sharer.basis_(coord).transpose();
      }
    });
  }
  static void UpdateGhostRhs(MyPart &part, std::vector<Coeff> *rhs) {
    assert(rhs->size() == part.CountLocalCells());
    part.ForEachGhostFace([&rhs](MyFace &face){
      auto& gauss = *(face.gauss_ptr_);
      auto& holder = *(face.holder_);
      auto& sharer = *(face.sharer_);
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        auto& coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.projection_(coord);
        Value u_sharer = sharer.projection_(coord);
        auto& riemann = face.GetRiemann(q);
        Value flux = riemann.GetFluxOnTimeAxis(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        rhs->at(holder.id()) -= flux * holder.basis_(coord).transpose();
      }
    });
  }
  static void UpdateBoundaryRhs(MyPart &part, std::vector<Coeff> *rhs) {
    part.ForEachSolidFace([&rhs](MyFace &face){
      auto& gauss = *(face.gauss_ptr_);
      assert(face.sharer_ == nullptr);
      auto& holder = *(face.holder_);
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        auto& coord = gauss.GetGlobalCoord(q);
        auto& riemann = face.GetRiemann(q);
        Value u_holder = holder.projection_(coord);
        Value flux = riemann.GetFluxOnSolidWall(u_holder);
        flux *= gauss.GetGlobalWeight(q);
        // std::cout << holder.gauss_ptr_->volume() << ' ' << holder.volume() << ' ' << holder.basis_(coord) << std::endl;
        rhs->at(holder.id()) -= flux * holder.basis_(coord).transpose();
      }
    });
  }
  void InitializeRhs(const MyPart &part) {
    rhs_.resize(part.CountLocalCells());
    for (auto& coeff : rhs_) {
      coeff.setZero();
    }
    part.ForEachLocalCell([&](const MyCell &cell){
      auto& r = rhs_.at(cell.id());
      auto& gauss = *(cell.gauss_ptr_);
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        auto& xyz = gauss.GetGlobalCoord(q);
        Value cv = cell.projection_(xyz);
        auto flux = Gas::GetFluxMatrix(cv);
        auto grad = cell.basis_.GetGradValue(xyz);
        Coeff prod = flux * grad.transpose();
        prod *= gauss.GetGlobalWeight(q);
        r += prod;
      }
    });
  }
  void UpdateLocalRhs(MyPart &part) {
    UpdateLocalRhs(part, &rhs_);
  }
  void UpdateGhostRhs(MyPart &part) {
    UpdateGhostRhs(part, &rhs_);
  }
  void UpdateBoundaryRhs(MyPart &part) {
    UpdateBoundaryRhs(part, &rhs_);
  }
  void SolveFrac13() {
    auto n_cells = u_old_.size();
    assert(n_cells == rhs_.size());
    u_frac13_ = rhs_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_frac13_[i_cell] *= dt_;
      u_frac13_[i_cell] += u_old_[i_cell];
    }
  }
  void SolveFrac23() {
    auto n_cells = u_old_.size();
    assert(n_cells == rhs_.size());
    assert(n_cells == u_frac13_.size());
    u_frac23_ = rhs_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_frac23_[i_cell] *= dt_;
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
    u_new_ = rhs_;
    for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
      u_new_[i_cell] *= dt_;
      u_new_[i_cell] += u_frac23_[i_cell];
      u_new_[i_cell] += u_frac23_[i_cell];
      u_new_[i_cell] += u_old_[i_cell];
      u_new_[i_cell] /= 3;
    }
  }
};
