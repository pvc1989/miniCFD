// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_FR_LOBATTO_HPP_
#define MINI_SPATIAL_FR_LOBATTO_HPP_

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

#include "mini/gauss/lobatto.hpp"
#include "mini/spatial/fr/general.hpp"
#include "mini/basis/vincent.hpp"

namespace mini {
namespace spatial {
namespace fr {

/**
 * @brief A specialized version of FR using a Lagrange expansion on Lobatto roots with the "Lumping Lobatto" correction function.
 * 
 * The \f$ g_\mathrm{right} \f$ only corrects the flux divergence at the rightest solution point, which is a flux point.
 * 
 * @tparam Part 
 */
template <typename Part>
class Lobatto : public General<Part> {
 public:
  using Base = General<Part>;
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
  using Vincent = typename Base::Vincent;

 protected:
  using FluxMatrix = typename Riemann::FluxMatrix;
  using GaussOnCell = typename Projection::Gauss;
  using GaussOnLine = typename GaussOnCell::GaussX;
  static_assert(std::is_same_v<GaussOnLine, typename GaussOnCell::GaussY>);
  static_assert(std::is_same_v<GaussOnLine, typename GaussOnCell::GaussZ>);
  static constexpr int kLineQ = GaussOnLine::Q;
  static_assert(std::is_same_v<GaussOnLine,
      mini::gauss::Lobatto<Scalar, kLineQ>>);
  static constexpr int kFaceQ = kLineQ * kLineQ;
  static constexpr int kCellQ = kLineQ * kFaceQ;

  struct FluxPointCache {
    Global normal;  // normal_flux = normal * flux_matrix
    Scalar scale;  // riemann_flux_local = scale * riemann_flux_global
    Scalar g_prime;
    int ijk;
  };
  using FaceCache = std::array<FluxPointCache, kFaceQ>;
  std::vector<FaceCache> holder_cache_;
  std::vector<FaceCache> sharer_cache_;

  static bool Collinear(Global const &a, Global const &b) {
    return std::abs(1 - std::abs(a.dot(b) / a.norm() / b.norm())) < 1e-8;
  }

  template <std::ranges::input_range R, class FaceToCell>
  void CacheCorrectionGradients(R &&faces, FaceToCell &&face_to_cell,
      std::vector<FaceCache> *cache) {
    Scalar g_prime = this->vincent_.LocalToRightDerivative(1);
    for (const Face &face : faces) {
      assert(cache->size() == face.id());
      auto &curr_face = cache->emplace_back();
      const auto &face_gauss = face.gauss();
      const auto &cell = face_to_cell(face);
      const auto &cell_gauss = cell.gauss();
      const auto &cell_basis = cell.basis();
      const auto &cell_projection = cell.projection();
      int i_face = cell_projection.FindFaceId(face.lagrange().center());
      for (int f = 0, F = face_gauss.CountPoints(); f < F; ++f) {
        Global const &face_normal = face_gauss.GetNormalFrame(f)[0];
        auto &flux_point = curr_face.at(f);
        auto &flux_point_coord = face_gauss.GetGlobalCoord(f);
        auto [i, j, k] = cell_projection.FindCollinearIndex(flux_point_coord, i_face);
        switch (i_face) {
        case 0:
          assert(k == -1);
          flux_point.ijk = cell_basis.index(i, j, 0);
          assert(Near(flux_point_coord, cell_gauss.GetGlobalCoord(flux_point.ijk)));
          flux_point.normal =
              cell_projection.GetJacobianAssociated(flux_point.ijk).col(Z);
          assert(Collinear(face_normal, flux_point.normal));
          flux_point.scale = -flux_point.normal.norm();
          flux_point.g_prime = -g_prime;
          break;
        case 1:
          assert(j == -1);
          flux_point.ijk = cell_basis.index(i, 0, k);
          assert(Near(flux_point_coord, cell_gauss.GetGlobalCoord(flux_point.ijk)));
          flux_point.normal =
              cell_projection.GetJacobianAssociated(flux_point.ijk).col(Y);
          assert(Collinear(face_normal, flux_point.normal));
          flux_point.scale = -flux_point.normal.norm();
          flux_point.g_prime = -g_prime;
          break;
        case 2:
          assert(i == -1);
          flux_point.ijk = cell_basis.index(GaussOnLine::Q - 1, j, k);
          assert(Near(flux_point_coord, cell_gauss.GetGlobalCoord(flux_point.ijk)));
          flux_point.normal =
              cell_projection.GetJacobianAssociated(flux_point.ijk).col(X);
          assert(Collinear(face_normal, flux_point.normal));
          flux_point.scale = +flux_point.normal.norm();
          flux_point.g_prime = +g_prime;
          break;
        case 3:
          assert(j == -1);
          flux_point.ijk = cell_basis.index(i, GaussOnLine::Q - 1, k);
          assert(Near(flux_point_coord, cell_gauss.GetGlobalCoord(flux_point.ijk)));
          flux_point.normal =
              cell_projection.GetJacobianAssociated(flux_point.ijk).col(Y);
          assert(Collinear(face_normal, flux_point.normal));
          flux_point.scale = +flux_point.normal.norm();
          flux_point.g_prime = +g_prime;
          break;
        case 4:
          assert(i == -1);
          flux_point.ijk = cell_basis.index(0, j, k);
          assert(Near(flux_point_coord, cell_gauss.GetGlobalCoord(flux_point.ijk)));
          flux_point.normal =
              cell_projection.GetJacobianAssociated(flux_point.ijk).col(X);
          assert(Collinear(face_normal, flux_point.normal));
          flux_point.scale = -flux_point.normal.norm();
          flux_point.g_prime = -g_prime;
          break;
        case 5:
          assert(k == -1);
          flux_point.ijk = cell_basis.index(i, j, GaussOnLine::Q - 1);
          assert(Near(flux_point_coord, cell_gauss.GetGlobalCoord(flux_point.ijk)));
          flux_point.normal =
              cell_projection.GetJacobianAssociated(flux_point.ijk).col(Z);
          assert(Collinear(face_normal, flux_point.normal));
          flux_point.scale = +flux_point.normal.norm();
          flux_point.g_prime = +g_prime;
          break;
        default:
          assert(false);
        }
      }
    }
  }

 public:
  explicit Lobatto(Part *part_ptr)
      : Base(part_ptr, Vincent::HuynhLumpingLobatto(Part::kDegrees)) {
    // TODO(PVC): remove duplicated code
    auto face_to_holder = [](auto &face) -> auto & { return face.holder(); };
    auto face_to_sharer = [](auto &face) -> auto & { return face.sharer(); };
    auto local_faces = this->part_ptr_->GetLocalFaces();
    CacheCorrectionGradients(local_faces, face_to_holder, &holder_cache_);
    CacheCorrectionGradients(local_faces, face_to_sharer, &sharer_cache_);
    auto ghost_faces = this->part_ptr_->GetGhostFaces();
    CacheCorrectionGradients(ghost_faces, face_to_holder, &holder_cache_);
    CacheCorrectionGradients(ghost_faces, face_to_sharer, &sharer_cache_);
    auto boundary_faces = this->part_ptr_->GetBoundaryFaces();
    CacheCorrectionGradients(boundary_faces, face_to_holder, &holder_cache_);
  }
  Lobatto(const Lobatto &) = default;
  Lobatto &operator=(const Lobatto &) = default;
  Lobatto(Lobatto &&) noexcept = default;
  Lobatto &operator=(Lobatto &&) noexcept = default;
  ~Lobatto() noexcept = default;

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
        auto &holder_flux_point = holder_cache[f];
        auto &sharer_flux_point = sharer_cache[f];
        auto [f_holder, f_sharer] = Base::GetFluxOnLocalFace(face, f,
            holder.projection(), holder_flux_point,
            sharer.projection(), sharer_flux_point);
        Projection::MinusValue(holder_flux_point.g_prime * f_holder,
                  holder_data, holder_flux_point.ijk);
        Projection::MinusValue(sharer_flux_point.g_prime * f_sharer,
                  sharer_data, sharer_flux_point.ijk);
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
        auto &holder_flux_point = holder_cache[f];
        auto &sharer_flux_point = sharer_cache[f];
        auto [f_holder, _] = Base::GetFluxOnLocalFace(face, f,
            holder.projection(), holder_flux_point,
            sharer.projection(), sharer_flux_point);
        Projection::MinusValue(holder_flux_point.g_prime * f_holder,
                  holder_data, holder_flux_point.ijk);
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
          auto &holder_flux_point = holder_cache[f];
          Value u_holder = holder.projection().GetValue(
              holder_flux_point.ijk);
          Value f_upwind = face.riemann(f).GetFluxOnSolidWall(u_holder);
          Value f_holder = f_upwind * holder_flux_point.scale -
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          Projection::MinusValue(holder_flux_point.g_prime * f_holder,
                    holder_data, holder_flux_point.ijk);
        }
      }
    }
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
          auto &holder_flux_point = holder_cache[f];
          Value u_holder = holder.projection().GetValue(
              holder_flux_point.ijk);
          Value f_upwind = face.riemann(f).GetFluxOnSupersonicOutlet(u_holder);
          Value f_holder = f_upwind * holder_flux_point.scale -
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          Projection::MinusValue(holder_flux_point.g_prime * f_holder,
                    holder_data, holder_flux_point.ijk);
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
          auto &holder_flux_point = holder_cache[f];
          auto f_holder = Base::GetFluxOnSupersonicFace(face, f, holder.projection(), holder_flux_point);
          Projection::MinusValue(holder_flux_point.g_prime * f_holder,
                    holder_data, holder_flux_point.ijk);
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
          auto &holder_flux_point = holder_cache[f];
          Value u_holder = holder.projection().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value f_upwind = face.riemann(f).GetFluxOnSubsonicInlet(u_holder, u_given);
          Value f_holder = f_upwind * holder_flux_point.scale -
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          Projection::MinusValue(holder_flux_point.g_prime * f_holder,
                    holder_data, holder_flux_point.ijk);
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
          auto &holder_flux_point = holder_cache[f];
          Value u_holder = holder.projection().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value f_upwind = face.riemann(f).GetFluxOnSubsonicOutlet(u_holder, u_given);
          Value f_holder = f_upwind * holder_flux_point.scale -
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          Projection::MinusValue(holder_flux_point.g_prime * f_holder,
                    holder_data, holder_flux_point.ijk);
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
          auto &holder_flux_point = holder_cache[f];
          Value u_holder = holder.projection().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(gauss.GetGlobalCoord(f), this->t_curr_);
          Value f_upwind = face.riemann(f).GetFluxOnSmartBoundary(u_holder, u_given);
          Value f_holder = f_upwind * holder_flux_point.scale -
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          Projection::MinusValue(holder_flux_point.g_prime * f_holder,
                    holder_data, holder_flux_point.ijk);
        }
      }
    }
  }
};

}  // namespace fr
}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_FR_LOBATTO_HPP_
