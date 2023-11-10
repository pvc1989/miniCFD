//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/mesh/vtk.hpp"
#include "mini/limiter/weno.hpp"
#include "mini/riemann/rotated/multiple.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/spatial/dg.hpp"

int n_core, i_core;
double time_begin;

constexpr int kComponents{2}, kDimensions{3}, kDegrees{2};
using Scalar = double;
using Riemann = mini::
    riemann::rotated::Multiple<Scalar, kComponents, kDimensions>;
using Coord = typename Riemann::Vector;
using Value = typename Riemann::Conservative;
Value func(const Coord& xyz) {
  auto r = std::hypot(xyz[0] - 2, xyz[1] - 0.5);
  return Value(r, 1 - r + (r >= 1));
}

// mpirun -n 4 ./part must be run in ../mesh
// mpirun -n 4 ./dg
int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &n_core);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  auto case_name = std::string("../mesh/double_mach");

  time_begin = MPI_Wtime();

  /* aproximated by Projection on OrthoNormal basis */
{
  using Projection = mini::polynomial::Projection<
      Scalar, kDimensions, kDegrees, kComponents>;
  using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
  auto part = Part(case_name, i_core, n_core);
  using Cell = typename Part::Cell;
  using Limiter = mini::limiter::weno::Lazy<Cell>;
  auto limiter = Limiter(/* w0 = */0.001, /* eps = */1e-6);
  using Spatial = mini::spatial::DiscontinuousGalerkin<Part, Limiter>;
  auto spatial = Spatial(&part, limiter);
  for (Cell *cell_ptr : part.GetLocalCellPointers()) {
    cell_ptr->Approximate(func);
  }
  part.Reconstruct(limiter);
  std::printf("Start test on proc[%d/%d] at %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  spatial.SetTime(1.5);
  auto column = spatial.GetSolutionColumn();
  assert(column.size() == part.GetCellDataSize());
  spatial.SetSolutionColumn(column);
  column -= spatial.GetSolutionColumn();
  std::printf("column.norm() == %6.2e on proc[%d/%d] at %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
}
  MPI_Finalize();
}
