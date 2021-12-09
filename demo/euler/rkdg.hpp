//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef DEMO_EULER_HPP_
#define DEMO_EULER_HPP_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/part.hpp"

namespace mini {
namespace mesh {
namespace cgns {

extern template class Part<cgsize_t, double, 5, 3, 0>;
extern template class Part<cgsize_t, double, 5, 3, 2>;

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // DEMO_EULER_HPP_
