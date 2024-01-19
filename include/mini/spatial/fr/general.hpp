// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_FR_GENERAL_HPP_
#define MINI_SPATIAL_FR_GENERAL_HPP_

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
#include <utility>

#include "mini/riemann/concept.hpp"
#include "mini/spatial/fem.hpp"
#include "mini/basis/vincent.hpp"

namespace mini {
namespace spatial {
namespace fr {

/**
 * @brief A general version of FR using a Lagrange expansion whose flux points are also solution points.
 * 
 * @tparam Part 
 */
template <typename Part>
class General : public spatial::FiniteElement<Part> {
 public:
  using Base = spatial::FiniteElement<Part>;
  using Riemann = typename Base::Riemann;
  using Scalar = typename Base::Scalar;
  using Face = typename Base::Face;
  using Cell = typename Base::Cell;
  using Global = typename Base::Global;
  using Projection = typename Base::Projection;
  static_assert(Projection::kLocal);
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

  struct SolutionPointCache {
    Scalar g_prime;
    int ijk;
  };
  struct FluxPointCache {
    Global normal;  // normal_flux = flux_matrix * normal
    Scalar scale;  // riemann_flux_local = scale * riemann_flux_global
    int ijk;
  };
  using LineCache = std::pair<
      std::array<SolutionPointCache, kLineQ>,
      FluxPointCache
  >;

  using FaceCache = std::array<LineCache, kFaceQ>;
  std::vector<FaceCache> holder_cache_;
  std::vector<FaceCache> sharer_cache_;

  using Vincent = mini::basis::Vincent<Scalar>;
  Vincent vincent_;

  static bool Collinear(Global const &a, Global const &b) {
    return std::abs(1 - std::abs(a.dot(b) / a.norm() / b.norm())) < 1e-8;
  }

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
        Global const &face_normal = face_gauss.GetNormalFrame(f)[0];
        auto &[curr_line, flux_point] = curr_face.at(f);
        auto &flux_point_coord = face_gauss.GetGlobalCoord(f);
        auto [i, j, k] = cell_projection.FindCollinearIndex(flux_point_coord, i_face);
        switch (i_face) {
        case 0:
          assert(k == -1);
          flux_point.ijk = cell_basis.index(i, j, 0);
          assert(Near(flux_point_coord, cell_gauss.GetGlobalCoord(flux_point.ijk)));
          flux_point.normal =
              cell_projection.GetJacobianAssociated(flux_point.ijk).col(Z);
          flux_point.scale = -flux_point.normal.norm();
          assert(Collinear(face_normal, flux_point.normal));
          for (k = 0; k < GaussOnLine::Q; ++k) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_gauss.GetLocalCoord(ijk);
            auto g_prime = vincent_.LocalToLeftDerivative(local[Z]);
            curr_line[k].g_prime = g_prime;
            curr_line[k].ijk = ijk;
          }
          break;
        case 1:
          assert(j == -1);
          flux_point.ijk = cell_basis.index(i, 0, k);
          assert(Near(flux_point_coord, cell_gauss.GetGlobalCoord(flux_point.ijk)));
          flux_point.normal =
              cell_projection.GetJacobianAssociated(flux_point.ijk).col(Y);
          flux_point.scale = -flux_point.normal.norm();
          assert(Collinear(face_normal, flux_point.normal));
          for (j = 0; j < GaussOnLine::Q; ++j) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_gauss.GetLocalCoord(ijk);
            auto g_prime = vincent_.LocalToLeftDerivative(local[Y]);
            curr_line[j].g_prime = g_prime;
            curr_line[j].ijk = ijk;
          }
          break;
        case 2:
          assert(i == -1);
          flux_point.ijk = cell_basis.index(GaussOnLine::Q - 1, j, k);
          assert(Near(flux_point_coord, cell_gauss.GetGlobalCoord(flux_point.ijk)));
          flux_point.normal =
              cell_projection.GetJacobianAssociated(flux_point.ijk).col(X);
          flux_point.scale = +flux_point.normal.norm();
          assert(Collinear(face_normal, flux_point.normal));
          for (i = 0; i < GaussOnLine::Q; ++i) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_gauss.GetLocalCoord(ijk);
            auto g_prime = vincent_.LocalToRightDerivative(local[X]);
            curr_line[i].g_prime = g_prime;
            curr_line[i].ijk = ijk;
          }
          break;
        case 3:
          assert(j == -1);
          flux_point.ijk = cell_basis.index(i, GaussOnLine::Q - 1, k);
          assert(Near(flux_point_coord, cell_gauss.GetGlobalCoord(flux_point.ijk)));
          flux_point.normal =
              cell_projection.GetJacobianAssociated(flux_point.ijk).col(Y);
          flux_point.scale = +flux_point.normal.norm();
          assert(Collinear(face_normal, flux_point.normal));
          for (j = 0; j < GaussOnLine::Q; ++j) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_gauss.GetLocalCoord(ijk);
            auto g_prime = vincent_.LocalToRightDerivative(local[Y]);
            curr_line[j].g_prime = g_prime;
            curr_line[j].ijk = ijk;
          }
          break;
        case 4:
          assert(i == -1);
          flux_point.ijk = cell_basis.index(0, j, k);
          assert(Near(flux_point_coord, cell_gauss.GetGlobalCoord(flux_point.ijk)));
          flux_point.normal =
              cell_projection.GetJacobianAssociated(flux_point.ijk).col(X);
          flux_point.scale = -flux_point.normal.norm();
          assert(Collinear(face_normal, flux_point.normal));
          for (i = 0; i < GaussOnLine::Q; ++i) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_gauss.GetLocalCoord(ijk);
            auto g_prime = vincent_.LocalToLeftDerivative(local[X]);
            curr_line[i].g_prime = g_prime;
            curr_line[i].ijk = ijk;
          }
          break;
        case 5:
          assert(k == -1);
          flux_point.ijk = cell_basis.index(i, j, GaussOnLine::Q - 1);
          assert(Near(flux_point_coord, cell_gauss.GetGlobalCoord(flux_point.ijk)));
          flux_point.normal =
              cell_projection.GetJacobianAssociated(flux_point.ijk).col(Z);
          flux_point.scale = +flux_point.normal.norm();
          assert(Collinear(face_normal, flux_point.normal));
          for (k = 0; k < GaussOnLine::Q; ++k) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_gauss.GetLocalCoord(ijk);
            auto g_prime = vincent_.LocalToRightDerivative(local[Z]);
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
  General(Part *part_ptr, Scalar c_next)
      : Base(part_ptr), vincent_(Part::kDegrees, c_next) {
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
  General(const General &) = default;
  General &operator=(const General &) = default;
  General(General &&) noexcept = default;
  General &operator=(General &&) noexcept = default;
  ~General() noexcept = default;

 protected:  // override virtual methods defined in Base
  void AddFluxDivergence(Column *residual) const override {
    for (const Cell &cell : this->part_ptr_->GetLocalCells()) {
      auto i_cell = cell.id();
      auto *data = residual->data() + this->part_ptr_->GetCellDataOffset(i_cell);
      const auto &gauss = cell.gauss();
      std::array<FluxMatrix, kCellQ> flux;
      for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
        FluxMatrix global_flux = Base::GetFluxMatrix(cell.projection(), q);
        flux[q] = cell.projection().GlobalFluxToLocalFlux(global_flux, q);
      }
      for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
        auto const &grad = cell.projection().GetBasisGradients(q);
        Value value = flux[0] * grad.col(0);
        for (int k = 1; k < n; ++k) {
          value += flux[k] * grad.col(k);
        }
        Projection::MinusValue(value, data, q);
      }
    }
  }
  template <typename Cache>
  static std::pair<Value, Value> GetFluxOnLocalFace(Face const &face, int f,
      const Projection &holder_projection, Cache const &holder_cache,
      const Projection &sharer_projection, Cache const &sharer_cache)
      requires(!mini::riemann::Diffusive<Riemann>) {
    Riemann const &riemann = face.riemann(f);
    Value u_holder = holder_projection.GetValue(holder_cache.ijk);
    Value u_sharer = sharer_projection.GetValue(sharer_cache.ijk);
    Value f_upwind = riemann.GetFluxUpwind(u_holder, u_sharer);
    assert(Collinear(holder_cache.normal, sharer_cache.normal));
    Value f_holder = f_upwind * holder_cache.scale -
        Riemann::GetFluxMatrix(u_holder) * holder_cache.normal;
    Value f_sharer = f_upwind * (-sharer_cache.scale) -
        Riemann::GetFluxMatrix(u_sharer) * sharer_cache.normal;
    return { f_holder, f_sharer };
  }
  template <typename Cache>
  static std::pair<Value, Value> GetFluxOnLocalFace(Face const &face, int f,
      const Projection &holder_projection, Cache const &holder_cache,
      const Projection &sharer_projection, Cache const &sharer_cache)
      requires(mini::riemann::ConvectiveDiffusive<Riemann>) {
    Riemann const &riemann = face.riemann(f);
    Value u_holder = holder_projection.GetValue(holder_cache.ijk);
    Value u_sharer = sharer_projection.GetValue(sharer_cache.ijk);
    Value f_upwind = riemann.GetFluxUpwind(u_holder, u_sharer);
    auto du_holder = holder_projection.GetGlobalGradient(holder_cache.ijk);
    auto du_sharer = sharer_projection.GetGlobalGradient(sharer_cache.ijk);
    auto ddu_holder = holder_projection.GetGlobalHessian(holder_cache.ijk);
    auto ddu_sharer = sharer_projection.GetGlobalHessian(sharer_cache.ijk);
    assert(Collinear(holder_cache.normal, sharer_cache.normal));
    const auto &normal = riemann.normal();
    auto distance = normal.dot(face.HolderToSharer());
    assert(distance > 0);
    auto du_common = riemann.GetCommonGradient(distance, normal,
        u_holder, u_sharer, du_holder, du_sharer, ddu_holder, ddu_sharer);
    Value u_common = (u_holder + u_sharer) / 2;
    Riemann::ModifyCommonFlux(u_common, du_common, normal, &f_upwind);
    auto f_mat_holder = Riemann::GetFluxMatrix(u_holder);
    Riemann::ModifyFluxMatrix(u_holder, du_holder, &f_mat_holder);
    Value f_holder = f_upwind * holder_cache.scale -
        f_mat_holder * holder_cache.normal;
    auto f_mat_sharer = Riemann::GetFluxMatrix(u_sharer);
    Riemann::ModifyFluxMatrix(u_sharer, du_sharer, &f_mat_sharer);
    Value f_sharer = f_upwind * (-sharer_cache.scale) -
        f_mat_sharer * sharer_cache.normal;
    return { f_holder, f_sharer };
  }
  void AddFluxOnLocalFaces(Column *residual) const override {
    for (const Face &face : this->part_ptr_->GetLocalFaces()) {
      const auto &gauss = face.gauss();
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      auto *holder_data = residual->data()
          + this->part_ptr_->GetCellDataOffset(holder.id());
      auto *sharer_data = residual->data()
          + this->part_ptr_->GetCellDataOffset(sharer.id());
      auto &holder_cache = holder_cache_[face.id()];
      auto &sharer_cache = sharer_cache_[face.id()];
      for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
        auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
        auto &[sharer_solution_points, sharer_flux_point] = sharer_cache[f];
        auto [f_holder, f_sharer] = GetFluxOnLocalFace(face, f,
            holder.projection(), holder_flux_point,
            sharer.projection(), sharer_flux_point);
        for (auto [g_prime, ijk] : holder_solution_points) {
          Projection::MinusValue(f_holder * g_prime, holder_data, ijk);
        }
        for (auto [g_prime, ijk] : sharer_solution_points) {
          Projection::MinusValue(f_sharer * g_prime, sharer_data, ijk);
        }
      }
    }
  }
  void AddFluxOnGhostFaces(Column *residual) const override {
    for (const Face &face : this->part_ptr_->GetGhostFaces()) {
      const auto &gauss = face.gauss();
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      auto *holder_data = residual->data()
          + this->part_ptr_->GetCellDataOffset(holder.id());
      auto &holder_cache = holder_cache_[face.id()];
      auto &sharer_cache = sharer_cache_[face.id()];
      for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
        auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
        auto &[sharer_solution_points, sharer_flux_point] = sharer_cache[f];
        auto [f_holder, _] = GetFluxOnLocalFace(face, f,
            holder.projection(), holder_flux_point,
            sharer.projection(), sharer_flux_point);
        for (auto [g_prime, ijk] : holder_solution_points) {
          Projection::MinusValue(f_holder * g_prime, holder_data, ijk);
        }
      }
    }
  }
  void ApplySolidWall(Column *residual) const override {
    for (const auto &name : this->solid_wall_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        auto &holder_cache = holder_cache_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
          Value u_holder = holder.projection().GetValue(
              holder_flux_point.ijk);
          Value f_upwind = face.riemann(f).GetFluxOnSolidWall(u_holder);
          Value f_holder = f_upwind * holder_flux_point.scale -
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          for (auto [g_prime, ijk] : holder_solution_points) {
            Projection::MinusValue(f_holder * g_prime, holder_data, ijk);
          }
        }
      }
    }
  }
  template <typename Cache>
  static Value GetFluxOnSupersonicFace(Face const &face, int f,
      const Projection &holder_projection, Cache const &holder_cache)
      requires(!mini::riemann::Diffusive<Riemann>) {
    Riemann const &riemann = face.riemann(f);
    Value u_holder = holder_projection.GetValue(holder_cache.ijk);
    Value f_upwind = riemann.GetFluxOnSupersonicOutlet(u_holder);
    auto f_mat_holder = Riemann::GetFluxMatrix(u_holder);
    Value f_holder = f_upwind * holder_cache.scale -
        f_mat_holder * holder_cache.normal;
    return f_holder;
  }
  template <typename Cache>
  static Value GetFluxOnSupersonicFace(Face const &face, int f,
      const Projection &holder_projection, Cache const &holder_cache)
      requires(mini::riemann::ConvectiveDiffusive<Riemann>) {
    Riemann const &riemann = face.riemann(f);
    Value u_holder = holder_projection.GetValue(holder_cache.ijk);
    Value f_upwind = riemann.GetFluxOnSupersonicOutlet(u_holder);
    auto du_holder = holder_projection.GetGlobalGradient(holder_cache.ijk);
    const auto &normal = riemann.normal();
    Riemann::ModifyCommonFlux(u_holder, du_holder, normal, &f_upwind);
    auto f_mat_holder = Riemann::GetFluxMatrix(u_holder);
    Riemann::ModifyFluxMatrix(u_holder, du_holder, &f_mat_holder);
    Value f_holder = f_upwind * holder_cache.scale -
        f_mat_holder * holder_cache.normal;
    return f_holder;
  }
  void ApplySupersonicOutlet(Column *residual) const override {
    for (const auto &name : this->supersonic_outlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        auto &holder_cache = holder_cache_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
          auto f_holder = GetFluxOnSupersonicFace(face, f, holder.projection(), holder_flux_point);
          for (auto [g_prime, ijk] : holder_solution_points) {
            Projection::MinusValue(f_holder * g_prime, holder_data, ijk);
          }
        }
      }
    }
  }
  void ApplySupersonicInlet(Column *residual) const override {
    for (auto &[name, func] : this->supersonic_inlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        auto &holder_cache = holder_cache_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
          Value u_holder = holder.projection().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value f_upwind = face.riemann(f).GetFluxOnSupersonicInlet(u_given);
          Value f_holder = f_upwind * holder_flux_point.scale -
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          for (auto [g_prime, ijk] : holder_solution_points) {
            Projection::MinusValue(f_holder * g_prime, holder_data, ijk);
          }
        }
      }
    }
  }
  void ApplySubsonicInlet(Column *residual) const override {
    for (auto &[name, func] : this->subsonic_inlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        auto &holder_cache = holder_cache_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
          Value u_holder = holder.projection().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value f_upwind = face.riemann(f).GetFluxOnSubsonicInlet(u_holder, u_given);
          Value f_holder = f_upwind * holder_flux_point.scale -
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          for (auto [g_prime, ijk] : holder_solution_points) {
            Projection::MinusValue(f_holder * g_prime, holder_data, ijk);
          }
        }
      }
    }
  }
  void ApplySubsonicOutlet(Column *residual) const override {
    for (auto &[name, func] : this->subsonic_outlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        auto &holder_cache = holder_cache_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
          Value u_holder = holder.projection().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value f_upwind = face.riemann(f).GetFluxOnSubsonicOutlet(u_holder, u_given);
          Value f_holder = f_upwind * holder_flux_point.scale -
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          for (auto [g_prime, ijk] : holder_solution_points) {
            Projection::MinusValue(f_holder * g_prime, holder_data, ijk);
          }
        }
      }
    }
  }
  void ApplySmartBoundary(Column *residual) const override {
    for (auto &[name, func] : this->smart_boundary_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        auto &holder_cache = holder_cache_[face.id()];
        for (int f = 0, n = gauss.CountPoints(); f < n; ++f) {
          auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
          Value u_holder = holder.projection().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value f_upwind = face.riemann(f).GetFluxOnSmartBoundary(u_holder, u_given);
          Value f_holder = f_upwind * holder_flux_point.scale -
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          for (auto [g_prime, ijk] : holder_solution_points) {
            Projection::MinusValue(f_holder * g_prime, holder_data, ijk);
          }
        }
      }
    }
  }
};

}  // namespace fr
}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_FR_GENERAL_HPP_
