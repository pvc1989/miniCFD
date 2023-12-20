// Copyright 2023 PEI Weicheng
#ifndef MINI_CONSTANT_INDEX_HPP_
#define MINI_CONSTANT_INDEX_HPP_

namespace mini {
namespace constant {
namespace index {

static constexpr int A{0};
static constexpr int B{1};
static constexpr int C{2};
static constexpr int D{3};

static constexpr int X{0};
static constexpr int Y{1};
static constexpr int Z{2};

static constexpr int XX{0};
static constexpr int XY{1}; static constexpr int YX{XY};
static constexpr int XZ{2}; static constexpr int ZX{XZ};
static constexpr int YY{3};
static constexpr int YZ{4}; static constexpr int ZY{YZ};
static constexpr int ZZ{5};

}  // namespace mini
}  // namespace constant
}  // namespace index

#endif  // MINI_CONSTANT_INDEX_HPP_
