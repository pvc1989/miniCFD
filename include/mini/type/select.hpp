// Copyright 2023 PEI Weicheng
#ifndef MINI_TYPE_SELECT_HPP_
#define MINI_TYPE_SELECT_HPP_

#include <concepts>

namespace mini {
namespace type {

/**
 * @brief A K-way type selection mechanism that extends `std::conditional_t`.
 * 
 */
// generic version, no instantiation:
template<unsigned N, typename... Types>
struct select;
// specialization for N > 0:
template <unsigned N, typename T, typename... Types>
struct select<N, T, Types...> {
  using type = typename select<N-1, Types...>::type;
};
// specialization for N == 0:
template <typename T, typename... Types>
struct select<0, T, Types...> {
  using type = T;
};
// STL-style type aliasing:
template<unsigned N, typename... Types>
using select_t = typename select<N, Types...>::type;

}  // namespace type
}  // namespace mini

#endif  // MINI_TYPE_SELECT_HPP_
