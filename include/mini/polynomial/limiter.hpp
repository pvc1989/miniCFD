//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_POLYNOMIAL_LIMITER_HPP_
#define MINI_POLYNOMIAL_LIMITER_HPP_

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace mini {
namespace polynomial {

template <typename Cell>
class LazyWeno {
  using Scalar = typename Cell::Scalar;
  using Projection = typename Cell::Projection;
  using Basis = typename Projection::Basis;
  using Coord = typename Projection::Coord;
  using Value = typename Projection::Value;

  std::vector<Projection> old_projections_;
  Projection *new_projection_ptr_ = nullptr;
  const Cell *my_cell_ = nullptr;
  Value weights_;
  Scalar eps_;
  bool verbose_;

 public:
  LazyWeno(Scalar w0, Scalar eps, bool verbose = false)
      : eps_(eps), verbose_(verbose) {
    weights_.setOnes();
    weights_ *= w0;
  }
  bool IsNotSmooth(const Cell &cell) {
    return true;
  }
  Projection operator()(const Cell &cell) {
    my_cell_ = &cell;
    Borrow();
    Reconstruct();
    assert(new_projection_ptr_);
    return *new_projection_ptr_;
  }
  void operator()(Cell *cell_ptr) {
    assert(cell_ptr);
    cell_ptr->projection_ = operator()(*cell_ptr);
  }

 private:
   /**
   * @brief Borrow projections from adjacent cells.
   * 
   */
  void Borrow() {
    old_projections_.clear();
    old_projections_.reserve(my_cell_->adj_cells_.size() + 1);
    auto my_average = my_cell_->projection_.GetAverage();
    for (auto *adj_cell : my_cell_->adj_cells_) {
      assert(adj_cell);
      old_projections_.emplace_back(adj_cell->projection_, my_cell_->basis_);
      auto &adj_proj = old_projections_.back();
      adj_proj += my_average - adj_proj.GetAverage();
      if (verbose_) {
        std::cout << "\n  adj smoothness[" << adj_cell->metis_id << "] = ";
        std::cout << std::scientific << std::setprecision(3) <<
            adj_proj.GetSmoothness().transpose();
      }
    }
    old_projections_.emplace_back(my_cell_->projection_);
    if (verbose_) {
      std::cout << "\n  old smoothness[" << my_cell_->metis_id << "] = ";
      std::cout << std::scientific << std::setprecision(3) <<
          old_projections_.back().GetSmoothness().transpose();
    }
    new_projection_ptr_ = &(old_projections_.back());
  }
  void Reconstruct() {
    int adj_cnt = my_cell_->adj_cells_.size();
    // initialize weights
    auto weights = std::vector<Value>(adj_cnt + 1, weights_);
    weights.back() *= -adj_cnt;
    weights.back().array() += 1.0;
    // modify weights by smoothness
    for (int i = 0; i <= adj_cnt; ++i) {
      auto &projection_i = old_projections_[i];
      auto beta = projection_i.GetSmoothness();
      beta.array() += eps_;
      beta.array() *= beta.array();
      weights[i].array() /= beta.array();
    }
    // normalize these weights
    Value sum; sum.setZero();
    sum = std::accumulate(weights.begin(), weights.end(), sum);
    assert(weights.size() == adj_cnt + 1);
    for (auto &weight : weights) {
      weight.array() /= sum.array();
    }
    // build the new (weighted) projection
    auto &new_projection = old_projections_.back();
    new_projection *= weights.back();
    for (int i = 0; i < adj_cnt; ++i) {
      old_projections_[i] *= weights[i];
      new_projection += old_projections_[i];
    }
  }
};

template <typename Cell>
class EigenWeno {
  using Scalar = typename Cell::Scalar;
  using Projection = typename Cell::Projection;
  using Face = typename Cell::Face;
  using Basis = typename Projection::Basis;
  using Coord = typename Projection::Coord;
  using Value = typename Projection::Value;

  Projection new_projection_;
  std::vector<Projection> old_projections_;
  const Cell *my_cell_ = nullptr;
  Value weights_;
  Scalar eps_;
  Scalar total_volume_;

 public:
  EigenWeno(Scalar w0, Scalar eps)
      : eps_(eps) {
    weights_.setOnes();
    weights_ *= w0;
  }
  static bool IsNotSmooth(const Cell &cell) {
    constexpr int kComponent = 0;
    auto center_value = cell.projection_(cell.center())[kComponent];
    auto average_max = std::max(1e-9,
        std::abs(cell.projection_.GetAverage()[kComponent]));
    auto difference_sum = 0.0;
    for (const Cell *adj_cell : cell.adj_cells_) {
      difference_sum += std::abs(center_value
          - adj_cell->projection_(cell.center())[kComponent]);
      average_max = std::max(average_max,
          std::abs(adj_cell->projection_.GetAverage()[kComponent]));
    }
    constexpr auto volume_power = (cell.kDegrees+1.0) / 2.0 / cell.kDimensions;
    auto smoothness = difference_sum / average_max
        / cell.adj_cells_.size() / std::pow(cell.volume(), volume_power);
    constexpr auto smoothness_reference = cell.kDegrees < 3 ? 1.0 : 3.0;
    return smoothness > smoothness_reference;
  }
  Projection operator()(const Cell &cell) {
    my_cell_ = &cell;
    Borrow();
    Reconstruct();
    return new_projection_;
  }
  void operator()(Cell *cell_ptr) {
    assert(cell_ptr);
    cell_ptr->projection_ = operator()(*cell_ptr);
  }

 private:
  /**
   * @brief Borrow projections from adjacent cells.
   * 
   */
  void Borrow() {
    old_projections_.clear();
    old_projections_.reserve(my_cell_->adj_cells_.size() + 1);
    auto my_average = my_cell_->projection_.GetAverage();
    for (auto *adj_cell : my_cell_->adj_cells_) {
      old_projections_.emplace_back(adj_cell->projection_, my_cell_->basis_);
      auto &adj_proj = old_projections_.back();
      adj_proj += my_average - adj_proj.GetAverage();
    }
    old_projections_.emplace_back(my_cell_->projection_);
  }
  /**
   * @brief Rotate borrowed projections onto the interface between cells
   * 
   */
  void ReconstructOnFace(const Face &adj_face) {
    assert(my_cell_->adj_faces_.size() == my_cell_->adj_cells_.size());
    int adj_cnt = my_cell_->adj_faces_.size();
    // build eigen-matrices in the rotated coordinate system
    const auto &big_u = my_cell_->projection_.GetAverage();
    auto *riemann = const_cast<typename Face::Riemann *>(&adj_face.riemann_);
    riemann->UpdateEigenMatrices(big_u);
    // initialize weights
    auto weights = std::vector<Value>(adj_cnt + 1, weights_);
    weights.back() *= -adj_cnt;
    weights.back().array() += 1.0;
    // modify weights by smoothness
    auto rotated_projections = old_projections_;
    for (int i = 0; i <= adj_cnt; ++i) {
      auto &projection_i = rotated_projections[i];
      projection_i.LeftMultiply(riemann->L());
      auto beta = projection_i.GetSmoothness();
      beta.array() += eps_;
      beta.array() *= beta.array();
      weights[i].array() /= beta.array();
    }
    // normalize these weights
    Value sum; sum.setZero();
    sum = std::accumulate(weights.begin(), weights.end(), sum);
    assert(weights.size() == adj_cnt + 1);
    for (auto &weight : weights) {
      weight.array() /= sum.array();
    }
    // build the new (weighted) projection
    auto &new_projection = rotated_projections.back();
    new_projection *= weights.back();
    for (int i = 0; i < adj_cnt; ++i) {
      rotated_projections[i] *= weights[i];
      new_projection += rotated_projections[i];
    }
    // rotate the new projection back to the global system
    new_projection.LeftMultiply(riemann->R());
    // scale the new projection by volume
    auto adj_volume = adj_face.other(my_cell_)->volume();
    new_projection *= adj_volume;
    new_projection_ += new_projection;
    total_volume_ += adj_volume;
  }
  /**
   * @brief Reconstruct projections by weights
   * 
   */
  void Reconstruct() {
    new_projection_ = Projection(my_cell_->basis_);
    total_volume_ = 0.0;
    for (auto *adj_face : my_cell_->adj_faces_) {
      ReconstructOnFace(*adj_face);
    }
    new_projection_ /= total_volume_;
  }
};

template <typename Cell>
class DummyWeno {
  using Scalar = typename Cell::Scalar;
  using Projection = typename Cell::Projection;

 public:
  DummyWeno(Scalar w0, Scalar eps, bool verbose = false) {
  }
  bool IsNotSmooth(const Cell &cell) {
    return true;
  }
  Projection operator()(const Cell &cell) {
    return cell.projection_;
  }
  void operator()(Cell *cell_ptr) {
  }
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_LIMITER_HPP_
