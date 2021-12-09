//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include "rkdg.hpp"

namespace mini {
namespace mesh {
namespace cgns {

template class Part<cgsize_t, double, 5, 3, 0>;
template class Part<cgsize_t, double, 5, 3, 2>;

}  // namespace cgns
}  // namespace mesh
}  // namespace mini
