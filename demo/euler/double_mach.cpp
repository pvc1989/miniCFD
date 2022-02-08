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
#include "mini/stepping/explicit.hpp"
#include "rkdg.hpp"

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int n_cores, i_core;
  MPI_Comm_size(MPI_COMM_WORLD, &n_cores);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  if (argc < 7) {
    if (i_core == 0) {
      std::cout << "usage:\n"
          << "  mpirun -n <n_cores> ./double_mach <cgns_file> <hexa|tetra>"
          << " <t_start> <t_stop> <n_steps_per_frame> <n_frames>"
          << " [<i_frame_start> [n_parts_prev]]\n";
    }
    MPI_Finalize();
    exit(0);
  }
  auto old_file_name = std::string(argv[1]);
  auto suffix = std::string(argv[2]);
  double t_start = std::atof(argv[3]);
  double t_stop = std::atof(argv[4]);
  int n_steps_per_frame = std::atoi(argv[5]);
  int n_frames = std::atoi(argv[6]);
  int n_steps = n_frames * n_steps_per_frame;
  auto dt = (t_stop - t_start) / n_steps;
  int i_frame = 0;
  if (argc > 7) {
    i_frame = std::atoi(argv[7]);
  }
  int n_parts_prev = 0;
  if (argc > 8) {
    n_parts_prev = std::atoi(argv[8]);
  }

  std::string case_name = "double_mach_" + suffix;

  auto time_begin = MPI_Wtime();

  /* Define the Euler system. */
  constexpr int kFunc = 5;
  constexpr int kDim = 3;
  using Primitive = mini::riemann::euler::PrimitiveTuple<double, kDim>;
  using Conservative = mini::riemann::euler::ConservativeTuple<double, kDim>;
  using Gas = mini::riemann::euler::IdealGas<double, 1, 4>;
  using Unrotated = mini::riemann::euler::Exact<Gas, kDim>;
  using Riemann = mini::riemann::rotated::Euler<Unrotated>;

  /* Partition the mesh. */
  if (i_core == 0 && n_parts_prev != n_cores) {
    using Shuffler = mini::mesh::Shuffler<idx_t, double>;
    Shuffler::PartitionAndShuffle(case_name, old_file_name, n_cores);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int kOrder = 2;
  using Part = mini::mesh::cgns::Part<cgsize_t, kOrder, Riemann>;
  using Cell = typename Part::Cell;
  using Face = typename Part::Face;
  using Coord = typename Cell::Coord;
  using Value = typename Cell::Value;
  using Coeff = typename Cell::Coeff;

  std::printf("Create a `Part` obj on proc[%d/%d] at %f sec\n",
      i_core, n_cores, MPI_Wtime() - time_begin);
  auto part = Part(case_name, i_core);
  part.SetFieldNames({"Density", "MomentumX", "MomentumY", "MomentumZ",
      "EnergyStagnationDensity"});

  /* Build a `Limiter` object. */
  // using Limiter = mini::polynomial::LazyWeno<Cell>;
  using Limiter = mini::polynomial::EigenWeno<Cell>;
  auto limiter = Limiter(/* w0 = */0.001, /* eps = */1e-6);

  /* Set initial conditions. */
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
  Value value_after = Gas::PrimitiveToConservative(primitive_after);
  Value value_before = Gas::PrimitiveToConservative(primitive_before);
  double x_gap = 1.0 / 6.0;
  auto initial_condition = [&](const Coord& xyz){
    auto x = xyz[0], y = xyz[1];
    return ((x - x_gap) * tan_60 < y) ? value_after : value_before;
  };

  if (argc == 7) {
    std::printf("Initialize by `Project()` on proc[%d/%d] at %f sec\n",
        i_core, n_cores, MPI_Wtime() - time_begin);
    part.ForEachLocalCell([&](Cell *cell_ptr){
      cell_ptr->Project(initial_condition);
    });

    std::printf("Run Reconstruct() on proc[%d/%d] at %f sec\n",
        i_core, n_cores, MPI_Wtime() - time_begin);
    auto limiter = Limiter(/* w0 = */0.001, /* eps = */1e-6);
    if (kOrder > 0) {
      part.Reconstruct(limiter);
      if (suffix == "tetra") {
        part.Reconstruct(limiter);
      }
    }

    std::printf("Run WriteSolutions(Frame0) on proc[%d/%d] at %f sec\n",
        i_core, n_cores, MPI_Wtime() - time_begin);
    part.GatherSolutions();
    part.WriteSolutions("Frame0");
    part.WriteSolutionsOnCellCenters("Frame0");
  } else {
    std::printf("Run ReadSolutions(Frame%d) on proc[%d/%d] at %f sec\n",
        i_frame, i_core, n_cores, MPI_Wtime() - time_begin);
    part.ReadSolutions("Frame" + std::to_string(i_frame));
    part.ScatterSolutions();
  }

  /* Choose the time-stepping scheme. */
  constexpr int kSteps = std::min(3, kOrder + 1);
  auto rk = RungeKutta<kSteps, Part, Limiter>(dt, limiter);

  /* Set boundary conditions. */
  auto u_x = u_gamma / cos_30;
  auto moving_shock = [&](const Coord& xyz, double t){
    auto x = xyz[0], y = xyz[1];
    return ((x - (x_gap + u_x * t)) * tan_60 < y) ? value_after : value_before;
  };
  if (suffix == "tetra") {
    rk.SetPrescribedBC("3_S_27", moving_shock);  // Top
    rk.SetPrescribedBC("3_S_31", moving_shock);  // Left
    rk.SetSolidWallBC("3_S_1");   // Back
    rk.SetSolidWallBC("3_S_32");  // Front
    rk.SetSolidWallBC("3_S_19");  // Bottom
    rk.SetFreeOutletBC("3_S_23");  // Right
    rk.SetFreeOutletBC("3_S_15");  // Gap
  } else {
    assert(suffix == "hexa");
    rk.SetPrescribedBC("4_S_27", moving_shock);  // Top
    rk.SetPrescribedBC("4_S_31", moving_shock);  // Left
    rk.SetSolidWallBC("4_S_1");   // Back
    rk.SetSolidWallBC("4_S_32");  // Front
    rk.SetSolidWallBC("4_S_19");  // Bottom
    rk.SetFreeOutletBC("4_S_23");  // Right
    rk.SetFreeOutletBC("4_S_15");  // Gap
  }

  /* Main Loop */
  auto wtime_start = MPI_Wtime();
  for (int i_step = 1; i_step <= n_steps; ++i_step) {
    double t_curr = t_start + dt * (i_step - 1);
    rk.Update(&part, t_curr);

    auto wtime_curr = MPI_Wtime() - wtime_start;
    auto wtime_left = wtime_curr * (n_steps - i_step) / (i_step);
    std::printf("[Done] Update(Step%d/%d) on proc[%d/%d] at %fs (%fs to go)\n",
        i_step, n_steps, i_core, n_cores, wtime_curr, wtime_left);

    if (i_step % n_steps_per_frame == 0) {
      ++i_frame;
      std::printf("Run WriteSolutions(Frame%d) on proc[%d/%d] at %f sec\n",
          i_frame, i_core, n_cores, MPI_Wtime() - time_begin);
      part.GatherSolutions();
      auto frame_name = "Frame" + std::to_string(i_frame);
      if (i_step == n_steps)
        part.WriteSolutions(frame_name);
      part.WriteSolutionsOnCellCenters(frame_name);
    }
  }

  std::printf("rank = %d, time = [%f, %f], frame = [%d, %d], dt = %f\n",
      i_core, t_start, t_stop, i_frame - n_frames, i_frame, dt);

  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      i_core, n_cores, MPI_Wtime() - time_begin);
  MPI_Finalize();
}
