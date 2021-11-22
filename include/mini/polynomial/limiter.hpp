//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_POLYNOMIAL_LIMITER_HPP_
#define MINI_POLYNOMIAL_LIMITER_HPP_

#include <cmath>
#include <iomanip>
#include <iostream>
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
  Projection* new_projection_ptr_ = nullptr;
  const Cell* my_cell_ = nullptr;
  Value weights_;
  Scalar eps_;
  bool verbose_;

 public:
  LazyWeno(const Scalar &w0, const Scalar &eps, bool verbose = false)
      : eps_(eps), verbose_(verbose) {
    weights_.setOnes();
    weights_ *= w0;
  }
  Projection operator()(const Cell& cell) {
    my_cell_ = &cell;
    Borrow();
    Reconstruct();
    assert(new_projection_ptr_);
    return *new_projection_ptr_;
  }
  void operator()(Cell* cell_ptr) {
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
    for (auto* adj_cell : my_cell_->adj_cells_) {
      assert(adj_cell);
      old_projections_.emplace_back(adj_cell->projection_, my_cell_->basis_);
      auto& adj_proj = old_projections_.back();
      adj_proj += my_average - adj_proj.GetAverage();
      if (verbose_) {
        std::printf("\n  adj smoothness[%2d] = ", adj_cell->metis_id);
        std::cout << std::scientific << std::setprecision(3) <<
            adj_proj.GetSmoothness().transpose();
      }
    }
    old_projections_.emplace_back(my_cell_->projection_);
    if (verbose_) {
      std::printf("\n  old smoothness[%2d] = ", my_cell_->metis_id);
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
      auto& projection_i = old_projections_[i];
      auto beta = projection_i.GetSmoothness();
      beta.array() += eps_;
      beta.array() *= beta.array();
      weights[i].array() /= beta.array();
    }
    // normalize these weights
    Value sum; sum.setZero();
    sum = std::accumulate(weights.begin(), weights.end(), sum);
    assert(weights.size() == adj_cnt + 1);
    for (auto& weight : weights) {
      weight.array() /= sum.array();
    }
    // build the new (weighted) projection
    auto& new_projection = old_projections_.back();
    new_projection *= weights.back();
    for (int i = 0; i < adj_cnt; ++i) {
      old_projections_[i] *= weights[i];
      new_projection += old_projections_[i];
    }
  }
};

template <typename Cell, typename Eigen>
class EigenWeno {
  using Scalar = typename Cell::Scalar;
  using Projection = typename Cell::Projection;
  using Basis = typename Projection::Basis;
  using Coord = typename Projection::Coord;
  using Value = typename Projection::Value;

  Projection new_projection_;
  std::vector<Projection> old_projections_;
  const Cell* my_cell_ = nullptr;
  Value weights_;
  Scalar eps_;
  Scalar total_volume_;

 public:
  EigenWeno(const Scalar &w0, const Scalar &eps)
      : eps_(eps) {
    weights_.setOnes();
    weights_ *= w0;
  }
  Projection operator()(const Cell& cell) {
    my_cell_ = &cell;
    Borrow();
    Reconstruct();
    return new_projection_;
  }
  void operator()(Cell* cell_ptr) {
    assert(cell_ptr);
    cell_ptr->projection_ = operator()(*cell_ptr);
  }

 private:
  static Coord GetNu(Cell const &cell_i, Cell const &cell_j) {
    Coord nu = cell_i.center() - cell_j.center();
    nu /= std::hypot(nu[0], nu[1], nu[2]);
    return nu;
  }
  static void GetMuPi(Coord const &nu, Coord *mu, Coord *pi) {
    int id = 0;
    for (int i = 1; i < 3; ++i) {
      if (std::abs(nu[i]) < std::abs(nu[id])) {
        id = i;
      }
    }
    auto a = nu[0], b = nu[1], c = nu[2];
    switch (id) {
    case 0:
      *mu << 0.0, -c, b;
      *pi << (b * b + c * c), -(a * b), -(a * c);
      break;
    case 1:
      *mu << c, 0.0, -a;
      *pi << -(a * b), (a * a + c * c), -(b * c);
      break;
    case 2:
      *mu << -b, a, 0.0;
      *pi << -(a * c), -(b * c), (a * a + b * b);
      break;
    default:
      break;
    }
    *mu /= std::hypot((*mu)[0], (*mu)[1], (*mu)[2]);
    *pi /= std::hypot((*pi)[0], (*pi)[1], (*pi)[2]);
  }
  /**
   * @brief Borrow projections from adjacent cells.
   * 
   */
  void Borrow() {
    old_projections_.clear();
    old_projections_.reserve(my_cell_->adj_cells_.size() + 1);
    auto my_average = my_cell_->projection_.GetAverage();
    for (auto* adj_cell : my_cell_->adj_cells_) {
      old_projections_.emplace_back(adj_cell->projection_, my_cell_->basis_);
      auto& adj_proj = old_projections_.back();
      adj_proj += my_average - adj_proj.GetAverage();
    }
    old_projections_.emplace_back(my_cell_->projection_);
  }
  /**
   * @brief Rotate borrowed projections onto the interface between cells
   * 
   */
  void ReconstructOnEachFace(const Cell &adj_cell) {
    int adj_cnt = my_cell_->adj_cells_.size();
    // build eigen-matrices in the rotated coordinate system
    Coord nu = GetNu(*my_cell_, adj_cell), mu, pi;
    GetMuPi(nu, &mu, &pi);
    // assert(nu.cross(mu) == pi);
    auto rotated_eigen = Eigen(my_cell_->projection_.GetAverage(), nu, mu, pi);
    // initialize weights
    auto weights = std::vector<Value>(adj_cnt + 1, weights_);
    weights.back() *= -adj_cnt;
    weights.back().array() += 1.0;
    // modify weights by smoothness
    auto rotated_projections = old_projections_;
    for (int i = 0; i <= adj_cnt; ++i) {
      auto& projection_i = rotated_projections[i];
      projection_i.LeftMultiply(rotated_eigen.L);
      auto beta = projection_i.GetSmoothness();
      beta.array() += eps_;
      beta.array() *= beta.array();
      weights[i].array() /= beta.array();
    }
    // normalize these weights
    Value sum; sum.setZero();
    sum = std::accumulate(weights.begin(), weights.end(), sum);
    assert(weights.size() == adj_cnt + 1);
    for (auto& weight : weights) {
      weight.array() /= sum.array();
    }
    // build the new (weighted) projection
    auto& new_projection = rotated_projections.back();
    new_projection *= weights.back();
    for (int i = 0; i < adj_cnt; ++i) {
      rotated_projections[i] *= weights[i];
      new_projection += rotated_projections[i];
    }
    // rotate the new projection back to the global system
    new_projection.LeftMultiply(rotated_eigen.R);
    // scale the new projection by volume
    auto adj_volume = adj_cell.volume();
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
    for (auto* adj_cell : my_cell_->adj_cells_) {
      ReconstructOnEachFace(*adj_cell);
    }
    new_projection_ /= total_volume_;
  }
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_LIMITER_HPP_
