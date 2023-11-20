//  Copyright 2023 PEI Weicheng
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
#include "mini/polynomial/hexahedron.hpp"
#include "mini/spatial/fem.hpp"
#include "mini/spatial/sem.hpp"

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
Value moving(const Coord& xyz, double t) {
  auto x = xyz[0], y = xyz[1];
  return Value(x + y, x - y);
}

// mpirun -n 4 ./part must be run in ../mesh
// mpirun -n 4 ./fr
int main(int argc, char* argv[]) {
  int n_core, i_core;
  double time_begin;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &n_core);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  auto case_name = std::string("../mesh/double_mach");

  using Jacobi = typename Riemann::Jacobi;
  Riemann::global_coefficient[0] = Jacobi{ {3., 0.}, {0., 4.} };
  Riemann::global_coefficient[1] = Jacobi{ {5., 0.}, {0., 6.} };
  Riemann::global_coefficient[2] = Jacobi{ {7., 0.}, {0., 8.} };

  /* aproximated by Projection on Lagrange basis on Lobatto roots */
{
  time_begin = MPI_Wtime();
  using Gx = mini::gauss::Lobatto<Scalar, kDegrees + 1>;
  using Gy = mini::gauss::Lobatto<Scalar, kDegrees + 1>;
  using Gz = mini::gauss::Lobatto<Scalar, kDegrees + 1>;
  using Projection = mini::polynomial::Hexahedron<Gx, Gy, Gz, kComponents>;
  using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
  using Spatial = mini::spatial::sem::FluxReconstruction<Part>;
  auto part = Part(case_name, i_core, n_core);
  auto spatial = Spatial(&part);
  spatial.SetSmartBoundary("4_S_27", moving);  // Top
  spatial.SetSmartBoundary("4_S_31", moving);  // Left
  spatial.SetSolidWall("4_S_1");   // Back
  spatial.SetSubsonicInlet("4_S_32", moving);  // Front
  spatial.SetSupersonicInlet("4_S_19", moving);  // Bottom
  spatial.SetSubsonicOutlet("4_S_23", moving);  // Right
  spatial.SetSupersonicOutlet("4_S_15");  // Gap
  for (auto *cell_ptr : part.GetLocalCellPointers()) {
    cell_ptr->Approximate(func);
  }
  spatial.SetTime(1.5);
  std::printf("Part on basis::lagrange::Hexahedron proc[%d/%d] cost %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  auto column = spatial.GetSolutionColumn();
  assert(column.size() == part.GetCellDataSize());
  spatial.SetSolutionColumn(column);
  column -= spatial.GetSolutionColumn();
  std::printf("solution.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  column = spatial.GetResidualColumn();
  std::printf("residual.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);
}
  MPI_Finalize();
}
