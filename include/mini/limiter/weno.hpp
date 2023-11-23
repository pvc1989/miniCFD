//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_LIMITER_WENO_HPP_
#define MINI_LIMITER_WENO_HPP_

#include <cassert>
#include <cmath>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace mini {
namespace limiter {
namespace weno {

template <typename Projection>
auto GetSmoothness(const Projection &proj) {
  using Coeff = typename Projection::Coeff;
  using Global = typename Projection::Global;
  using Taylor = typename Projection::Taylor;
  auto mat_pdv_func = [&proj](Global const &xyz) {
    auto local = xyz; local -= proj.center();
    auto mat_pdv = Taylor::GetPdvValue(local, proj.coeff());
    mat_pdv = mat_pdv.cwiseProduct(mat_pdv);
    return mat_pdv;
  };
  auto integral = gauss::Integrate(mat_pdv_func, proj.gauss());
  auto volume = proj.basis().Measure();
  return Taylor::GetSmoothness(integral, volume);
}

template <typename Cell>
class Lazy {
  using Scalar = typename Cell::Scalar;
  using Projection = typename Cell::Projection;
  using ProjectionWrapper = typename Projection::Wrapper;
  using Basis = typename Projection::Basis;
  using Global = typename Projection::Global;
  using Value = typename Projection::Value;

  std::vector<ProjectionWrapper> old_projections_;
  ProjectionWrapper *new_projection_ptr_ = nullptr;
  const Cell *my_cell_ = nullptr;
  Value weights_;
  Scalar eps_;
  bool verbose_;

 public:
  Lazy(Scalar w0, Scalar eps, bool verbose = false)
      : eps_(eps), verbose_(verbose) {
    weights_.setOnes();
    weights_ *= w0;
  }
  bool IsNotSmooth(const Cell &cell) {
    return true;
  }
  ProjectionWrapper operator()(const Cell &cell) {
    my_cell_ = &cell;
    Borrow();
    Reconstruct();
    assert(new_projection_ptr_);
    return *new_projection_ptr_;
  }

 private:
   /**
   * @brief Borrow projections from adjacent cells.
   * 
   */
  void Borrow() {
    old_projections_.clear();
    old_projections_.reserve(my_cell_->adj_cells_.size() + 1);
    auto my_average = my_cell_->projection_.average();
    for (auto *adj_cell : my_cell_->adj_cells_) {
      assert(adj_cell);
      auto &adj_proj = old_projections_.emplace_back(my_cell_->basis());
      assert(&(adj_proj.basis()) == &(my_cell_->basis()));
      adj_proj.Approximate(adj_cell->projection_);
      adj_proj += my_average - adj_proj.average();
      if (verbose_) {
        std::cout << "\n  adj smoothness[" << adj_cell->metis_id << "] = ";
        std::cout << std::scientific << std::setprecision(3) <<
            GetSmoothness(adj_proj).transpose();
      }
    }
    old_projections_.emplace_back(my_cell_->projection_);
    if (verbose_) {
      std::cout << "\n  old smoothness[" << my_cell_->metis_id << "] = ";
      std::cout << std::scientific << std::setprecision(3) <<
          GetSmoothness(old_projections_.back()).transpose();
    }
    new_projection_ptr_ = &(old_projections_.back());
    assert(&(new_projection_ptr_->basis()) == &(my_cell_->basis()));
  }
  void Reconstruct() {
    int adj_cnt = my_cell_->adj_cells_.size();
    // initialize weights
    auto weights = std::vector<Value>(adj_cnt + 1, weights_);
    weights.back() *= -adj_cnt;
    weights.back().array() += 1.0;
    // modify weights by smoothness
    for (int i = 0; i <= adj_cnt; ++i) {
      auto beta = GetSmoothness(old_projections_[i]);
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
class Eigen {
  using Scalar = typename Cell::Scalar;
  using Projection = typename Cell::Projection;
  using ProjectionWrapper = typename Projection::Wrapper;
  using Face = typename Cell::Face;
  using Basis = typename Projection::Basis;
  using Global = typename Projection::Global;
  using Value = typename Projection::Value;

  ProjectionWrapper new_projection_;
  std::vector<ProjectionWrapper> old_projections_;
  const Cell *my_cell_ = nullptr;
  Value weights_;
  Scalar eps_;
  Scalar total_volume_;

 public:
  Eigen(Scalar w0, Scalar eps)
      : eps_(eps) {
    weights_.setOnes();
    weights_ *= w0;
  }
  static bool IsNotSmooth(const Cell &cell) {
    constexpr int components[] = { 0, Cell::K-1 };
    auto max_abs_averages = cell.projection_.average();
    for (int i : components) {
      max_abs_averages[i] = std::max(1e-9, std::abs(max_abs_averages[i]));
    }
    typename Cell::Value sum_abs_differences; sum_abs_differences.setZero();
    auto my_values = cell.GlobalToValue(cell.center());
    for (const Cell *adj_cell : cell.adj_cells_) {
      auto adj_values = adj_cell->GlobalToValue(cell.center());
      auto adj_averages = adj_cell->projection_.average();
      for (int i : components) {
        sum_abs_differences[i] += std::abs(my_values[i] - adj_values[i]);
        max_abs_averages[i] = std::max(max_abs_averages[i],
            std::abs(adj_averages[i]));
      }
    }
    constexpr auto volume_power = (Cell::P + 1.0) / 2.0 / Cell::D;
    auto divisor = std::pow(cell.volume(), volume_power);
    divisor *= cell.adj_cells_.size();
    constexpr auto smoothness_reference = Cell::P < 3 ? 1.0 : 3.0;
    for (int i : components) {
      auto smoothness = sum_abs_differences[i] / max_abs_averages[i] / divisor;
      if (smoothness > smoothness_reference) {
        return true;  // if any component is not smooth
      }
    }
    return false;  // if all components are smooth
  }
  ProjectionWrapper operator()(const Cell &cell) {
    my_cell_ = &cell;
    Borrow();
    Reconstruct();
    return new_projection_;
  }

 private:
  /**
   * @brief Borrow projections from adjacent cells.
   * 
   */
  void Borrow() {
    old_projections_.clear();
    old_projections_.reserve(my_cell_->adj_cells_.size() + 1);
    auto my_average = my_cell_->projection_.average();
    for (auto *adj_cell : my_cell_->adj_cells_) {
      auto &adj_proj = old_projections_.emplace_back(my_cell_->basis());
      adj_proj.Approximate(adj_cell->projection_);
      adj_proj += my_average - adj_proj.average();
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
    const auto &big_u = my_cell_->projection_.average();
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
      auto beta = GetSmoothness(projection_i);
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
    new_projection_ = ProjectionWrapper(my_cell_->basis());
    new_projection_.coeff().setZero();
    total_volume_ = 0.0;
    for (auto *adj_face : my_cell_->adj_faces_) {
      ReconstructOnFace(*adj_face);
    }
    new_projection_ /= total_volume_;
  }
};

template <typename Cell>
class Dummy {
  using Scalar = typename Cell::Scalar;
  using Projection = typename Cell::Projection;

 public:
  Dummy(Scalar w0, Scalar eps, bool verbose = false) {
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

}  // namespace weno
}  // namespace limiter
}  // namespace mini

#endif  // MINI_LIMITER_WENO_HPP_
