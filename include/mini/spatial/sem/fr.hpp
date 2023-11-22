// Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_SPATIAL_SEM_FR_HPP_
#define MINI_SPATIAL_SEM_FR_HPP_

#include <concepts>
#include <ranges>

#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "mini/spatial/fem.hpp"
#include "mini/basis/vincent.hpp"

namespace mini {
namespace spatial {
namespace sem {

/**
 * @brief A specialized version of FR using a Lagrange expansion on Gaussian quadrature points. 
 * 
 * @tparam Part 
 */
template <typename Part>
class FluxReconstruction : public spatial::FiniteElement<Part> {
 public:
  using Base = spatial::FiniteElement<Part>;
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

 protected:
  using FluxMatrix = typename Riemann::FluxMatrix;
  using GaussOnCell = typename Projection::Gauss;
  using GaussOnLine = typename GaussOnCell::GaussX;
  static_assert(std::is_same_v<GaussOnLine, typename GaussOnCell::GaussY>);
  static_assert(std::is_same_v<GaussOnLine, typename GaussOnCell::GaussZ>);
  static constexpr int kLineQ = GaussOnLine::Q;
  static constexpr int kFaceQ = kLineQ * kLineQ;
  static constexpr int kCellQ = kLineQ * kFaceQ;

  struct Cache {
    Scalar g_prime;
    int ijk;
  };

  using LineCache = std::array<Cache, kLineQ>;
  using FaceCache = std::array<std::pair<LineCache, int>, kFaceQ>;
  std::vector<FaceCache> holder_cache_;
  std::vector<FaceCache> sharer_cache_;

  using Vincent = mini::basis::Vincent<Scalar>;
  Vincent vincent_;

  static constexpr int X = 0;
  static constexpr int Y = 1;
  static constexpr int Z = 2;

  template <std::ranges::input_range R, class FaceToCell>
  void CacheCorrectionGradients(R &&faces, FaceToCell &&face_to_cell,
      std::vector<FaceCache> *cache) {
    for (const Face &face : faces) {
      assert(cache->size() == face.id());
      auto &curr_face = cache->emplace_back();
      const auto &face_gauss = face.gauss();
      const auto &cell = face_to_cell(face);
      const auto &cell_lagrange = cell.lagrange();
      const auto &cell_gauss = cell.gauss();
      const auto &cell_basis = cell.basis();
      const auto &cell_projection = cell.projection();
      int i_face = cell_projection.FindFaceId(face.lagrange().center());
      for (int f = 0, F = face_gauss.CountPoints(); f < F; ++f) {
        auto &[curr_line, flux_point_ijk] = curr_face.at(f);
        auto &flux_point = face_gauss.GetGlobalCoord(f);
        auto [i, j, k] = cell_projection.FindCollinearIndex(flux_point, i_face);
        switch (i_face) {
        case 0:
          assert(k == -1);
          flux_point_ijk = cell_basis.index(i, j, 0);
          assert(Near(flux_point, cell_gauss.GetGlobalCoord(flux_point_ijk)));
          for (k = 0; k < GaussOnLine::Q; ++k) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_gauss.GetLocalCoord(ijk);
            auto g_prime = vincent_.LocalToLeftDerivative(local[Z]);
            auto jacobian = cell_lagrange.LocalToJacobian(local);
            g_prime /= jacobian(Z, Z);
            curr_line[k].g_prime = g_prime;
            curr_line[k].ijk = ijk;
          }
          break;
        case 1:
          assert(j == -1);
          flux_point_ijk = cell_basis.index(i, 0, k);
          assert(Near(flux_point, cell_gauss.GetGlobalCoord(flux_point_ijk)));
          for (j = 0; j < GaussOnLine::Q; ++j) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_gauss.GetLocalCoord(ijk);
            auto g_prime = vincent_.LocalToLeftDerivative(local[Y]);
            auto jacobian = cell_lagrange.LocalToJacobian(local);
            g_prime /= jacobian(Y, Y);
            curr_line[j].g_prime = g_prime;
            curr_line[j].ijk = ijk;
          }
          break;
        case 2:
          assert(i == -1);
          flux_point_ijk = cell_basis.index(GaussOnLine::Q - 1, j, k);
          assert(Near(flux_point, cell_gauss.GetGlobalCoord(flux_point_ijk)));
          for (i = 0; i < GaussOnLine::Q; ++i) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_gauss.GetLocalCoord(ijk);
            auto g_prime = vincent_.LocalToRightDerivative(local[X]);
            auto jacobian = cell_lagrange.LocalToJacobian(local);
            g_prime /= jacobian(X, X);
            curr_line[i].g_prime = g_prime;
            curr_line[i].ijk = ijk;
          }
          break;
        case 3:
          assert(j == -1);
          flux_point_ijk = cell_basis.index(i, GaussOnLine::Q - 1, k);
          assert(Near(flux_point, cell_gauss.GetGlobalCoord(flux_point_ijk)));
          for (j = 0; j < GaussOnLine::Q; ++j) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_gauss.GetLocalCoord(ijk);
            auto g_prime = vincent_.LocalToRightDerivative(local[Y]);
            auto jacobian = cell_lagrange.LocalToJacobian(local);
            g_prime /= jacobian(Y, Y);
            curr_line[j].g_prime = g_prime;
            curr_line[j].ijk = ijk;
          }
          break;
        case 4:
          assert(i == -1);
          flux_point_ijk = cell_basis.index(0, j, k);
          assert(Near(flux_point, cell_gauss.GetGlobalCoord(flux_point_ijk)));
          for (i = 0; i < GaussOnLine::Q; ++i) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_gauss.GetLocalCoord(ijk);
            auto g_prime = vincent_.LocalToLeftDerivative(local[X]);
            auto jacobian = cell_lagrange.LocalToJacobian(local);
            g_prime /= jacobian(X, X);
            curr_line[i].g_prime = g_prime;
            curr_line[i].ijk = ijk;
          }
          break;
        case 5:
          assert(k == -1);
          flux_point_ijk = cell_basis.index(i, j, GaussOnLine::Q - 1);
          assert(Near(flux_point, cell_gauss.GetGlobalCoord(flux_point_ijk)));
          for (k = 0; k < GaussOnLine::Q; ++k) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_gauss.GetLocalCoord(ijk);
            auto g_prime = vincent_.LocalToRightDerivative(local[Z]);
            auto jacobian = cell_lagrange.LocalToJacobian(local);
            g_prime /= jacobian(Z, Z);
            curr_line[k].g_prime = g_prime;
            curr_line[k].ijk = ijk;
          }
          break;
        default:
          assert(false);
        }
      }
    }
  }

 public:
  FluxReconstruction(Part *part_ptr, Vincent const &vincent)
      : Base(part_ptr), vincent_(vincent) {
    auto face_to_holder = [](auto &face) -> auto & { return face.holder(); };
    auto face_to_sharer = [](auto &face) -> auto & { return face.sharer(); };
    auto local_cells = this->part_ptr_->GetLocalFaces();
    CacheCorrectionGradients(local_cells, face_to_holder, &holder_cache_);
    CacheCorrectionGradients(local_cells, face_to_sharer, &sharer_cache_);
    auto ghost_cells = this->part_ptr_->GetGhostFaces();
    CacheCorrectionGradients(ghost_cells, face_to_holder, &holder_cache_);
    CacheCorrectionGradients(ghost_cells, face_to_sharer, &sharer_cache_);
    auto boundary_cells = this->part_ptr_->GetBoundaryFaces();
    CacheCorrectionGradients(boundary_cells, face_to_holder, &holder_cache_);
  }
  FluxReconstruction(const FluxReconstruction &) = default;
  FluxReconstruction &operator=(const FluxReconstruction &) = default;
  FluxReconstruction(FluxReconstruction &&) noexcept = default;
  FluxReconstruction &operator=(FluxReconstruction &&) noexcept = default;
  ~FluxReconstruction() noexcept = default;

 protected:  // override virtual methods defined in Base
  void AddFluxDivergence(Column *residual) const override {
    for (const Cell &cell : this->part_ptr_->GetLocalCells()) {
      auto i_cell = cell.id();
      auto *data = residual->data() + this->part_ptr_->GetCellDataOffset(i_cell);
      const auto &gauss = cell.gauss();
      std::array<FluxMatrix, kCellQ> flux;
      for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
        auto const &value = cell.projection().GetValueOnGaussianPoint(q);
        flux[q] = Riemann::GetFluxMatrix(value);
      }
      for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
        auto const &grad = cell.projection().GetBasisGradientsOnGaussianPoint(q);
        Value value = flux[0] * grad.col(0);
        for (int k = 1; k < n; ++k) {
          value += flux[k] * grad.col(k);
        }
        Projection::AddValueTo(value, data, q);
      }
    }
  }
  void AddFluxOnLocalFaces(Column *residual) const override {
    for (const Face &face : this->part_ptr_->GetLocalFaces()) {
      const auto &face_gauss = face.gauss();
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      const auto &riemann = face.riemann();
      auto *holder_data = residual->data()
          + this->part_ptr_->GetCellDataOffset(holder.id());
      auto *sharer_data = residual->data()
          + this->part_ptr_->GetCellDataOffset(sharer.id());
      auto &holder_cache = holder_cache_[face.id()];
      auto &sharer_cache = sharer_cache_[face.id()];
      for (int f = 0, n = face_gauss.CountPoints(); f < n; ++f) {
        auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
        auto &[sharer_solution_points, sharer_flux_point] = sharer_cache[f];
        Value u_holder =
            holder.projection().GetValueOnGaussianPoint(holder_flux_point);
        Value u_sharer =
            sharer.projection().GetValueOnGaussianPoint(sharer_flux_point);
        Value f_upwind = riemann.GetFluxUpwind(u_holder, u_sharer);
        Global const &normal = face_gauss.GetNormalFrame(f)[0];
        assert(normal.dot(sharer.center() - holder.center()) > 0);
        Value f_holder = f_upwind - Riemann::GetFluxMatrix(u_holder) * normal;
        for (auto [g_prime, ijk] : holder_solution_points) {
          Projection::AddValueTo(f_holder * g_prime, holder_data, ijk);
        }
        Value f_sharer = Riemann::GetFluxMatrix(u_sharer) * normal - f_upwind;
        for (auto [g_prime, ijk] : sharer_solution_points) {
          Projection::AddValueTo(f_sharer * g_prime, sharer_data, ijk);
        }
      }
    }
  }
  void AddFluxOnGhostFaces(Column *residual) const override {
  }
  void ApplySolidWall(Column *residual) const override {
  }
  void ApplySupersonicInlet(Column *residual) const override {
  }
  void ApplySupersonicOutlet(Column *residual) const override {
  }
  void ApplySubsonicInlet(Column *residual) const override {
  }
  void ApplySubsonicOutlet(Column *residual) const override {
  }
  void ApplySmartBoundary(Column *residual) const override {
  }
};

}  // namespace sem
}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_SEM_FR_HPP_
