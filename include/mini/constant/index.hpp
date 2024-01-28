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

/**
 * @brief for indexing vectors
 * 
 */
static constexpr int X{0};
static constexpr int Y{1};
static constexpr int Z{2};

/**
 * @brief for indexing symmetric (rank-2) tensors
 * 
 */
static constexpr int XX{0};
static constexpr int XY{1}; static constexpr int YX{XY};
static constexpr int XZ{2}; static constexpr int ZX{XZ};
static constexpr int YY{3};
static constexpr int YZ{4}; static constexpr int ZY{YZ};
static constexpr int ZZ{5};

/**
 * @brief for indexing symmetric rank-3 tensors
 * 
 */
static constexpr int XXX{0};
static constexpr int XXY{1};
static constexpr int XXZ{2};
static constexpr int XYX{XXY};
static constexpr int XYY{3};
static constexpr int XYZ{4};
static constexpr int XZX{XXZ};
static constexpr int XZY{XYZ};
static constexpr int XZZ{5};
//
static constexpr int YXX{XXY};
static constexpr int YXY{XYY};
static constexpr int YXZ{XYZ};
static constexpr int YYX{XYY};
static constexpr int YYY{6};
static constexpr int YYZ{7};
static constexpr int YZX{XYZ};
static constexpr int YZY{YYZ};
static constexpr int YZZ{8};
//
static constexpr int ZXX{XXZ};
static constexpr int ZXY{XYZ};
static constexpr int ZXZ{XZZ};
static constexpr int ZYX{XYZ};
static constexpr int ZYY{YYZ};
static constexpr int ZYZ{YZZ};
static constexpr int ZZX{XZZ};
static constexpr int ZZY{YZZ};
static constexpr int ZZZ{9};
//

}  // namespace index
}  // namespace constant
}  // namespace mini

#endif  // MINI_CONSTANT_INDEX_HPP_
