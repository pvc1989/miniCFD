#include <cassert>
#include <vector>

#include "mini/mesh/part.hpp"

template <class Coeff, int kOrder>
class RungeKutta;

template <class Coeff>
struct RungeKutta<Coeff, 3> {
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

  static void ReadFromLocalCells(const Part &part, std::vector<Coeff> *coeffs) {
    coeffs->resize(part.CountLocalCells());
    part.ForEachLocalCell([&coeffs](const auto &cell){
      coeffs->at(cell.id()) = cell.projection_.coeff();
    });
  }
  static void WriteToLocalCells(const std::vector<Coeff> &coeffs, Part *part) {
    assert(coeffs.size() == part->CountLocalCells());
    part.ForEachLocalCell([&coeffs](auto &cell){
      cell.projection_.UpdateCoeffs(coeffs.at(cell.id()));
    });
  }
  static void UpdateLocalRhs(const Part &part, std::vector<Coeff> *rhs) {
    rhs->resize(CountLocalCells());
    part.ForEachLocalFace([&rhs](auto &face){
      auto& gauss = *(face.gauss_ptr_);
      auto& holder = *(face.holder_);
      auto& sharer = *(face.sharer_);
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        auto& coord = gauss.GetGlobalCoord(q);
        auto u_holder = holder.projection_(coord);
        auto u_sharer = sharer.projection_(coord);
        auto& riemann = face.GetRiemannSolver(q);
        auto flux = riemann.GetFluxOnTimeAxis(u_holder, u_sharer);
        auto& weight = gauss.GetGlobalWeight(q);
        flux *= weight;
        rhs->at(holder.id()) += flux;
        rhs->at(sharer.id()) -= flux;
      }
    });
  }
  static void UpdateGhostRhs(const Part &part, std::vector<Coeff> *rhs) {
    assert(rhs->size() == part.CountLocalCells());
    part.ForEachGhostFace([](MyFace &face){
      auto& gauss = *(face.gauss_ptr_);
      auto& holder = *(face.holder_);
      auto& sharer = *(face.sharer_);
      for (int q = 0; q < gauss.CountQuadPoints(); ++q) {
        auto& coord = gauss.GetGlobalCoord(q);
        auto u_holder = holder.projection_(coord);
        auto u_sharer = sharer.projection_(coord);
        auto& riemann = face.GetRiemannSolver(q);
        auto flux = riemann.GetFluxOnTimeAxis(u_holder, u_sharer);
        auto& weight = gauss.GetGlobalWeight(q);
        flux *= weight;
        rhs->at(holder.id()) += flux;
      }
    });
  }
  static void UpdateBoundaryRhs(const Part &part, std::vector<Coeff> *rhs) {
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
