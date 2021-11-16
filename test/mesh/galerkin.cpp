//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/eigen.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/integrator/ode.hpp"

// mpirun -n 4 ./galerkin
int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int comm_size, comm_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  cgp_mpi_comm(MPI_COMM_WORLD);

  double t_final = std::atof(argv[1]);
  int n_steps = std::atoi(argv[2]);
  int n_steps_io = std::atoi(argv[3]);
  auto dt = t_final / n_steps;
  std::printf("rank = %d, time = [0.0, %f], step = [0, %d], dt = %f\n", comm_rank, t_final, n_steps, dt);

  auto time_begin = MPI_Wtime();

  constexpr int kFunc = 1;
  constexpr int kDim = 3;
  constexpr int kOrder = 2;
  using MyPart = mini::mesh::cgns::Part<cgsize_t, double, kFunc, kDim, kOrder>;
  using MyCell = typename MyPart::CellType;
  using MyFace = typename MyPart::FaceType;
  using Coord = typename MyCell::Coord;
  using Value = typename MyCell::Value;
  using Coeff = typename MyCell::Coeff;

  /* Linear Advection Problem */
  using Limiter = mini::polynomial::LazyWeno<MyCell>;
  MyFace::Riemann::global_coefficient[0] = -10.0;  // must proceed `Part()`
  Value value_after{ 10 }, value_before{ -10 };
  auto initial_condition = [&](const Coord& xyz){
    return (xyz[0] > 3.0) ? value_after : value_before;
  };

  /* Double Mach Reflection Problem 
  using Primitive = mini::riemann::euler::PrimitiveTuple<3>;
  using Conservative = mini::riemann::euler::ConservativeTuple<3>;
  using Gas = mini::riemann::euler::IdealGas<1, 4>;
  using Matrices = mini::riemann::euler::EigenMatrices<double, Gas>;
  using Limiter = mini::polynomial::EigenWeno<MyCell, Matrices>;
  // prepare the states before and after the shock
  double rho_before = 1.4, p_before = 1.0;
  double m_before = 10.0, a_before = 1.0, u_gamma = m_before * a_before;
  double gamma_plus = 2.4, gamma = 1.4, gamma_minus = 0.4;
  auto rho_after = rho_before * (m_before * m_before * gamma_plus / 2.0)
      / (1.0 + m_before * m_before * gamma_minus / 2.0);
  assert(rho_after == 8.0);
  auto p_after = p_before * (m_before * m_before * gamma - gamma_minus / 2.0)
      / (gamma_plus / 2.0);
  // assert(p_after == 116.5);
  std::cout << "p_after = " << p_after << '\n';
  auto u_n_after = u_gamma * (rho_after - rho_before) / rho_after;
  // assert(u_n_after == 8.25);
  std::cout << "u_n_after = " << u_n_after << '\n';
  auto tan_60 = std::sqrt(3.0), cos_30 = tan_60 * 0.5, sin_30 = 0.5;
  auto u_after = u_n_after * cos_30, v_after = u_n_after * (-sin_30);
  // auto primitive_after = Primitive(rho_after, u_after, v_after, 0.0, p_after);
  // auto primitive_before = Primitive(rho_before, 0.0, 0.0, 0.0, p_before);
  auto primitive_after = Primitive(1.0, 0, 0, 0.0, 1);
  auto primitive_before = Primitive(0.125, 0.0, 0.0, 0.0, 0.1);
  auto consv_after = Gas::PrimitiveToConservative(primitive_after);
  auto consv_before = Gas::PrimitiveToConservative(primitive_before);
  Value value_after = { consv_after.mass, consv_after.momentum[0],
      consv_after.momentum[1], consv_after.momentum[2], consv_after.energy };
  Value value_before = { consv_before.mass, consv_before.momentum[0],
      consv_before.momentum[1], consv_before.momentum[2], consv_before.energy };
  double x_gap = 1.0 / 6.0;
  auto initial_condition = [&](const Coord& xyz){
    auto x = xyz[0], y = xyz[1];
    // return ((x - x_gap) * tan_60 < y) ? value_after : value_before;
    return (x > 2.0) ? value_after : value_before;
  }; */

  std::printf("Create a `Part` obj on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  auto part = MyPart("double_mach_hexa", comm_rank);

  std::printf("Initialize by `Project()` on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  part.ForEachLocalCell([&](MyCell &cell){
    cell.Project(initial_condition);
  });

  std::printf("Run Reconstruct() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  auto limiter = Limiter(/* w0 = */0.001, /* eps = */1e-6);
  part.Reconstruct(limiter);

  std::printf("Run Write(0) on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  part.GatherSolutions();
  part.WriteSolutions("Step0");
  part.WriteSolutionsOnCellCenters("Step0");

  using RK = RungeKutta<MyPart, 3>;
  auto rk = RK(dt);
  for (int i_step = 1; i_step <= n_steps; ++i_step) {
    std::printf("Run Solve(%d) on proc[%d/%d] at %f sec\n",
        i_step, comm_rank, comm_size, MPI_Wtime() - time_begin);

    RK::ReadFromLocalCells(part, &rk.u_old_);

    part.ShareGhostCellCoeffs();
    rk.InitializeRhs(part);
    rk.UpdateLocalRhs(part);
    rk.UpdateBoundaryRhs(part);
    part.UpdateGhostCellCoeffs();
    rk.UpdateGhostRhs(part);
    rk.SolveFrac13();
    RK::WriteToLocalCells(rk.u_frac13_, &part);
    part.Reconstruct(limiter);

    part.ShareGhostCellCoeffs();
    rk.InitializeRhs(part);
    rk.UpdateLocalRhs(part);
    rk.UpdateBoundaryRhs(part);
    part.UpdateGhostCellCoeffs();
    rk.UpdateGhostRhs(part);
    rk.SolveFrac23();
    RK::WriteToLocalCells(rk.u_frac23_, &part);
    part.Reconstruct(limiter);

    part.ShareGhostCellCoeffs();
    rk.InitializeRhs(part);
    rk.UpdateLocalRhs(part);
    rk.UpdateBoundaryRhs(part);
    part.UpdateGhostCellCoeffs();
    rk.UpdateGhostRhs(part);
    rk.SolveFrac33();
    RK::WriteToLocalCells(rk.u_new_, &part);
    part.Reconstruct(limiter);

    if (i_step % n_steps_io == 0) {
      std::printf("Run Write(%d) on proc[%d/%d] at %f sec\n",
          i_step, comm_rank, comm_size, MPI_Wtime() - time_begin);
      part.GatherSolutions();
      auto step_name = "Step" + std::to_string(i_step);
      part.WriteSolutions(step_name);
      part.WriteSolutionsOnCellCenters(step_name);
    }
  }

  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  MPI_Finalize();
}
