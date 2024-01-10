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
#include "mini/spatial/fem.hpp"
#include "mini/spatial/dg/general.hpp"
#include "mini/spatial/dg/lobatto.hpp"
#include "mini/temporal/ode.hpp"
#include "mini/input/path.hpp"  // defines PROJECT_BINARY_DIR

constexpr int kComponents{2}, kDimensions{3}, kDegrees{2};
using Scalar = double;

mini::temporal::Euler<Scalar> temporal;
double t_curr = 1.5, dt = 1e-3;

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

template <class Part>
Scalar Norm1(Part const &part){
  auto norm_1 = 0.0;
  for (auto &cell : part.GetLocalCells()) {
    for (int i = 0, n = cell.gauss().CountPoints(); i < n; ++i) {
      auto v = cell.projection().GetValue(i);
      norm_1 += std::abs(v[0]) + std::abs(v[1]);
    }
  }
  return norm_1;
}

auto case_name = PROJECT_BINARY_DIR + std::string("/test/mesh/double_mach");

// mpirun -n 4 ./part must be run in ../mesh
// mpirun -n 4 ./dg
int main(int argc, char* argv[]) {
  int n_core, i_core;
  double time_begin;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &n_core);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  using Jacobi = typename Riemann::Jacobi;
  Riemann::SetConvectionCoefficient(
    Jacobi{ {3., 0.}, {0., 4.} },
    Jacobi{ {5., 0.}, {0., 6.} },
    Jacobi{ {7., 0.}, {0., 8.} }
  );

  /* aproximated by Projection on OrthoNormal basis */
{
  time_begin = MPI_Wtime();
  using Projection = mini::polynomial::Projection<
      Scalar, kDimensions, kDegrees, kComponents>;
  using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
  auto part = Part(case_name, i_core, n_core);
  using Cell = typename Part::Cell;
  using Limiter = mini::limiter::weno::Lazy<Cell>;
  auto limiter = Limiter(/* w0 = */0.001, /* eps = */1e-6);
  using Spatial = mini::spatial::dg::WithLimiterAndSource<Part, Limiter>;
  auto spatial = Spatial(&part, limiter);
  spatial.SetSmartBoundary("4_S_27", moving);  // Top
  spatial.SetSmartBoundary("4_S_31", moving);  // Left
  spatial.SetSolidWall("4_S_1");   // Back
  spatial.SetSubsonicInlet("4_S_32", moving);  // Front
  spatial.SetSupersonicInlet("4_S_19", moving);  // Bottom
  spatial.SetSubsonicOutlet("4_S_23", moving);  // Right
  spatial.SetSupersonicOutlet("4_S_15");  // Gap
  for (Cell *cell_ptr : part.GetLocalCellPointers()) {
    cell_ptr->Approximate(func);
  }
  part.Reconstruct(limiter);
  spatial.SetTime(1.5);
  std::printf("Part on basis::OrthoNormal proc[%d/%d] cost %f sec\n",
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
  /* aproximated by Projection on Lagrange basis on Lobatto roots */
{
  time_begin = MPI_Wtime();
  using Gx = mini::gauss::Lobatto<Scalar, kDegrees + 1>;

  /* Check equivalence between local and global formulation. */
{
  using Projection = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents, true>;
  using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
  using Spatial = mini::spatial::dg::Lobatto<Part>;
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
  std::printf("Part on basis::lagrange::Hexahedron<Local> proc[%d/%d] cost %f sec\n",
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

  time_begin = MPI_Wtime();
  std::printf("val_curr.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      Norm1(part), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  temporal.Update(&spatial, t_curr, dt);
  std::printf("val_next.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      Norm1(part), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);
}
  using Projection = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents>;
  using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
  using Spatial = mini::spatial::dg::Lobatto<Part>;
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
  std::printf("Part on basis::lagrange::Hexahedron<Global> proc[%d/%d] cost %f sec\n",
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

  time_begin = MPI_Wtime();
  std::printf("val_curr.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      Norm1(part), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  temporal.Update(&spatial, t_curr, dt);
  std::printf("val_next.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      Norm1(part), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  std::printf("Check the consistency between FEM and SEM implementations.\n");
  MPI_Barrier(MPI_COMM_WORLD);
  class Test : public Spatial {
    using SEM = Spatial;
    using FEM = typename Spatial::Base;

   public:
    explicit Test(Part *part_ptr)
        : Spatial(part_ptr) {
    }

    void AddFluxDivergence(Column *residual) const override {
      residual->setZero();
      this->SEM::AddFluxDivergence(residual);
      *residual *= -1.0;
      this->FEM::AddFluxDivergence(residual);
    }
    void AddFluxOnLocalFaces(Column *residual) const override {
      residual->setZero();
      this->SEM::AddFluxOnLocalFaces(residual);
      *residual *= -1.0;
      this->FEM::AddFluxOnLocalFaces(residual);
    }
    void AddFluxOnGhostFaces(Column *residual) const override {
      residual->setZero();
      this->SEM::AddFluxOnGhostFaces(residual);
      *residual *= -1.0;
      this->FEM::AddFluxOnGhostFaces(residual);
    }
    void ApplySolidWall(Column *residual) const override {
      residual->setZero();
      this->SEM::ApplySolidWall(residual);
      *residual *= -1.0;
      this->FEM::ApplySolidWall(residual);
    }
    void ApplySupersonicInlet(Column *residual) const override {
      residual->setZero();
      this->SEM::ApplySupersonicInlet(residual);
      *residual *= -1.0;
      this->FEM::ApplySupersonicInlet(residual);
    }
    void ApplySupersonicOutlet(Column *residual) const override {
      residual->setZero();
      this->SEM::ApplySupersonicOutlet(residual);
      *residual *= -1.0;
      this->FEM::ApplySupersonicOutlet(residual);
    }
    void ApplySubsonicInlet(Column *residual) const override {
      residual->setZero();
      this->SEM::ApplySubsonicInlet(residual);
      *residual *= -1.0;
      this->FEM::ApplySubsonicInlet(residual);
    }
    void ApplySubsonicOutlet(Column *residual) const override {
      residual->setZero();
      this->SEM::ApplySubsonicOutlet(residual);
      *residual *= -1.0;
      this->FEM::ApplySubsonicOutlet(residual);
    }
    void ApplySmartBoundary(Column *residual) const override {
      residual->setZero();
      this->SEM::ApplySmartBoundary(residual);
      *residual *= -1.0;
      this->FEM::ApplySmartBoundary(residual);
    }
  };
  auto test = Test(&part);
  test.SetSmartBoundary("4_S_27", moving);  // Top
  test.SetSmartBoundary("4_S_31", moving);  // Left
  test.SetSolidWall("4_S_1");   // Back
  test.SetSubsonicInlet("4_S_32", moving);  // Front
  test.SetSupersonicInlet("4_S_19", moving);  // Bottom
  test.SetSubsonicOutlet("4_S_23", moving);  // Right
  test.SetSupersonicOutlet("4_S_15");  // Gap

  time_begin = MPI_Wtime();
  test.AddFluxDivergence(&column);
  std::printf("AddFluxDivergence.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.AddFluxOnLocalFaces(&column);
  std::printf("AddFluxOnLocalFaces.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.AddFluxOnGhostFaces(&column);
  std::printf("AddFluxOnGhostFaces.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.ApplySolidWall(&column);
  std::printf("ApplySolidWall.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);
  
  time_begin = MPI_Wtime();
  test.ApplySupersonicInlet(&column);
  std::printf("ApplySupersonicInlet.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);
  
  time_begin = MPI_Wtime();
  test.ApplySupersonicOutlet(&column);
  std::printf("ApplySupersonicOutlet.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.ApplySubsonicInlet(&column);
  std::printf("ApplySubsonicInlet.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.ApplySubsonicOutlet(&column);
  std::printf("ApplySubsonicOutlet.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.ApplySmartBoundary(&column);
  std::printf("ApplySmartBoundary.norm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.norm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);
}
  MPI_Finalize();
}
