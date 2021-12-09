//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/shuffler.hpp"
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/eigen.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/integrator/ode.hpp"
#include "rkdg.hpp"

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int n_procs, i_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_proc);
  cgp_mpi_comm(MPI_COMM_WORLD);

  if (argc < 7) {
    if (i_proc == 0) {
      std::cout << "usage:\n";
      std::cout << "  mpirun -n <n_proc> ./double_mach <cgns_file> <t_start>";
      std::cout << " <t_stop> <n_steps> <n_steps_per_frame> <hexa|tetra>\n";
    }
    MPI_Finalize();
    exit(0);
  }
  auto old_file_name = std::string(argv[1]);
  double t_start = std::atof(argv[2]);
  double t_stop = std::atof(argv[3]);
  int n_steps = std::atoi(argv[4]);
  int n_steps_per_frame = std::atoi(argv[5]);
  auto dt = t_stop / n_steps;
  auto suffix = std::string(argv[6]);

  std::string case_name = "double_mach_" + suffix;
  std::printf("rank = %d, time = [0.0, %f], step = [0, %d], dt = %f\n",
      i_proc, t_stop, n_steps, dt);

  auto time_begin = MPI_Wtime();

  /* Partition the mesh */
  if (i_proc == 0) {
    using MyShuffler = mini::mesh::Shuffler<idx_t, double>;
    MyShuffler::PartitionAndShuffle(case_name, old_file_name, n_procs);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int kFunc = 5;
  constexpr int kDim = 3;
  constexpr int kOrder = 2;
  constexpr int kTemporalAccuracy = std::min(3, kOrder + 1);
  using MyPart = mini::mesh::cgns::Part<cgsize_t, double, kFunc, kDim, kOrder>;
  using MyCell = typename MyPart::CellType;
  using MyFace = typename MyPart::FaceType;
  using Coord = typename MyCell::Coord;
  using Value = typename MyCell::Value;
  using Coeff = typename MyCell::Coeff;

  std::printf("Create a `Part` obj on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  auto part = MyPart(case_name, i_proc);

  /* Euler system */
  using Primitive = mini::riemann::euler::PrimitiveTuple<kDim>;
  using Conservative = mini::riemann::euler::ConservativeTuple<kDim>;
  using Gas = mini::riemann::euler::IdealGas<1, 4, double>;
  using Matrices = mini::riemann::euler::EigenMatrices<double, Gas>;
  using MyLimiter = mini::polynomial::EigenWeno<MyCell, Matrices>;
  using Unrotated = mini::riemann::euler::Exact<Gas, kDim>;
  using MyRiemann = mini::riemann::rotated::Euler<Unrotated>;
  // using MyLimiter = mini::polynomial::LazyWeno<MyCell>;

  /* IC for Double-Mach Problem */
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
  auto primitive_after = Primitive(rho_after, u_after, v_after, 0.0, p_after);
  auto primitive_before = Primitive(rho_before, 0.0, 0.0, 0.0, p_before);
  auto consv_after = Gas::PrimitiveToConservative(primitive_after);
  auto consv_before = Gas::PrimitiveToConservative(primitive_before);
  Value value_after = { consv_after.mass, consv_after.momentum[0],
      consv_after.momentum[1], consv_after.momentum[2], consv_after.energy };
  Value value_before = { consv_before.mass, consv_before.momentum[0],
      consv_before.momentum[1], consv_before.momentum[2], consv_before.energy };
  double x_gap = 1.0 / 6.0;
  auto initial_condition = [&](const Coord& xyz){
    auto x = xyz[0], y = xyz[1];
    return ((x - x_gap) * tan_60 < y) ? value_after : value_before;
  };

  std::printf("Initialize by `Project()` on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  part.ForEachLocalCell([&](MyCell *cell_ptr){
    cell_ptr->Project(initial_condition);
  });

  std::printf("Run Reconstruct() on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  auto limiter = MyLimiter(/* w0 = */0.001, /* eps = */1e-6);
  if (kOrder > 0) {
    part.Reconstruct(limiter);
    if (suffix == "tetra") {
      part.Reconstruct(limiter);
    }
  }

  std::printf("Run Write(Step0) on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  part.GatherSolutions();
  // part.WriteSolutions("Step0");
  part.WriteSolutionsOnCellCenters("Step0");

  auto rk = RungeKutta<kTemporalAccuracy, MyPart, MyRiemann>(dt);
  rk.BuildRiemannSolvers(part);

  /* Boundary Conditions */
  auto u_x = u_gamma / cos_30;
  auto moving_shock = [&](const Coord& xyz, double t){
    auto x = xyz[0], y = xyz[1];
    return ((x - (x_gap + u_x * t)) * tan_60 < y) ? value_after : value_before;
  };
  if (suffix == "tetra") {
    rk.SetPrescribedBC("3_S_27", moving_shock);  // Top
    rk.SetPrescribedBC("3_S_31", moving_shock);  // Left
    rk.SetSolidWallBC("3_S_27");  // Top
    rk.SetSolidWallBC("3_S_31");  // Left
    rk.SetSolidWallBC("3_S_1");   // Back
    rk.SetSolidWallBC("3_S_32");  // Front
    rk.SetSolidWallBC("3_S_19");  // Bottom
    rk.SetFreeOutletBC("3_S_23");  // Right
    rk.SetFreeOutletBC("3_S_15");  // Gap
  } else {
    assert(suffix == "hexa");
    rk.SetPrescribedBC("4_S_27", moving_shock);  // Top
    rk.SetPrescribedBC("4_S_31", moving_shock);  // Left
    rk.SetSolidWallBC("4_S_27");  // Top
    rk.SetSolidWallBC("4_S_31");  // Left
    rk.SetSolidWallBC("4_S_1");   // Back
    rk.SetSolidWallBC("4_S_32");  // Front
    rk.SetSolidWallBC("4_S_19");  // Bottom
    rk.SetFreeOutletBC("4_S_23");  // Right
    rk.SetFreeOutletBC("4_S_15");  // Gap
  }

  /* Main Loop */
  for (int i_step = 1; i_step <= n_steps; ++i_step) {
    std::printf("Run Update(Step%d) on proc[%d/%d] at %f sec\n",
        i_step, i_proc, n_procs, MPI_Wtime() - time_begin);
    double t_curr = t_start + dt * i_step;
    rk.Update(&part, t_curr, limiter);

    if (i_step % n_steps_per_frame == 0) {
      std::printf("Run Write(Step%d) on proc[%d/%d] at %f sec\n",
          i_step, i_proc, n_procs, MPI_Wtime() - time_begin);
      part.GatherSolutions();
      auto step_name = "Step" + std::to_string(i_step);
      if (i_step == n_steps)
        part.WriteSolutions(step_name);
      part.WriteSolutionsOnCellCenters(step_name);
    }
  }

  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  MPI_Finalize();
}
