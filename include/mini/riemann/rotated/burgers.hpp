#ifndef MINI_RIEMANN_ROTATED_BURGERS_HPP_
#define MINI_RIEMANN_ROTATED_BURGERS_HPP_

#include "mini/riemann/rotated/simple.hpp"
#include "mini/riemann/nonlinear/burgers.hpp"

namespace mini {
namespace riemann {
namespace rotated {

using Burgers = Simple<nonlinear::Burgers>;

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_ROTATED_BURGERS_HPP_
