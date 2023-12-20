//  Copyright 2023 PEI Weicheng

#ifndef MINI_RIEMANN_CONCEPT_HPP_
#define MINI_RIEMANN_CONCEPT_HPP_

#include <concepts>
#include <type_traits>

namespace mini {
namespace riemann {

template <typename R>
concept HasConvectiveData = requires {
  requires std::integral<decltype(R::kComponents)>;
  requires std::integral<decltype(R::kDimensions)>;

  typename R::Scalar;
  requires std::floating_point<typename R::Scalar>;

  typename R::Vector;
  typename R::Conservative;
  typename R::Flux;
  typename R::FluxMatrix;
};

template <typename R, typename C>
concept HasConvectiveMethods = requires(R r, C const &x, C const &y) {
  requires HasConvectiveData<R>;
  requires std::same_as<C, typename R::Conservative>;
  { R::GetFluxMatrix(x) } -> std::same_as<typename R::FluxMatrix>;
  { r.GetFluxUpwind(x, y) } -> std::same_as<typename R::Flux>;
};

template <typename R>
concept Convective =
    HasConvectiveData<R> &&
    HasConvectiveMethods<R, typename R::Conservative>;

}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_CONCEPT_HPP_
