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
  int n_procs, i_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_proc);
  cgp_mpi_comm(MPI_COMM_WORLD);

  if (argc < 7) {
    if (i_proc == 0) {
      std::cout << "usage:\n"
          << "  mpirun -n <n_proc> ./forward_step <cgns_file> <hexa|tetra>"
          << " <t_start> <t_stop> <n_steps> <n_steps_per_frame>"
          << " [<i_start> [n_parts_prev]]\n";
    }
    MPI_Finalize();
    exit(0);
  }
  auto old_file_name = std::string(argv[1]);
  auto suffix = std::string(argv[2]);
  double t_start = std::atof(argv[3]);
  double t_stop = std::atof(argv[4]);
  int n_steps = std::atoi(argv[5]);
  int n_steps_per_frame = std::atoi(argv[6]);
  auto dt = (t_stop - t_start) / n_steps;
  int i_start = 0;
  if (argc > 7) {
    i_start = std::atoi(argv[7]);
  }
  int n_parts_prev = 0;
  if (argc > 8) {
    n_parts_prev = std::atoi(argv[8]);
  }
  int i_stop = i_start + n_steps;

  std::string case_name = "forward_step_" + suffix;

  auto time_begin = MPI_Wtime();

  /* Partition the mesh */
  if (i_proc == 0 && n_parts_prev != n_procs) {
    using Shuffler = mini::mesh::Shuffler<idx_t, double>;
    Shuffler::PartitionAndShuffle(case_name, old_file_name, n_procs);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int kFunc = 5;
  constexpr int kDim = 3;
  constexpr int kOrder = 2;
  constexpr int kSteps = std::min(3, kOrder + 1);
  using Part = mini::mesh::cgns::Part<cgsize_t, double, kFunc, kDim, kOrder>;
  using Cell = typename Part::Cell;
  using Face = typename Part::Face;
  using Coord = typename Cell::Coord;
  using Value = typename Cell::Value;
  using Coeff = typename Cell::Coeff;

  std::printf("Create a `Part` obj on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  auto part = Part(case_name, i_proc);
  part.SetFieldNames({"Density", "MomentumX", "MomentumY", "MomentumZ",
      "EnergyStagnationDensity"});

  /* Euler system */
  using Primitive = mini::riemann::euler::PrimitiveTuple<double, kDim>;
  using Conservative = mini::riemann::euler::ConservativeTuple<double, kDim>;
  using Gas = mini::riemann::euler::IdealGas<double, 1, 4>;
  using Matrices = mini::riemann::euler::EigenMatrices<Gas>;
  using Limiter = mini::polynomial::EigenWeno<Cell, Matrices>;
  using Unrotated = mini::riemann::euler::Exact<Gas, kDim>;
  using Riemann = mini::riemann::rotated::Euler<Unrotated>;
  // using Limiter = mini::polynomial::LazyWeno<Cell>;
  auto limiter = Limiter(/* w0 = */0.001, /* eps = */1e-6);

  /* IC for Forward-Step Problem */
  auto primitive = Primitive(1.4, 3.0, 0.0, 0.0, 1.0);
  Value given_value = Gas::PrimitiveToConservative(primitive);
  auto initial_condition = [&given_value](const Coord& xyz){
    return given_value;
  };

  if (argc == 7) {
    std::printf("Initialize by `Project()` on proc[%d/%d] at %f sec\n",
        i_proc, n_procs, MPI_Wtime() - time_begin);
    part.ForEachLocalCell([&](Cell *cell_ptr){
      cell_ptr->Project(initial_condition);
    });

    std::printf("Run Reconstruct() on proc[%d/%d] at %f sec\n",
        i_proc, n_procs, MPI_Wtime() - time_begin);
    if (kOrder > 0) {
      part.Reconstruct(limiter);
      if (suffix == "tetra") {
        part.Reconstruct(limiter);
      }
    }

    std::printf("Run WriteSolutions(Step0) on proc[%d/%d] at %f sec\n",
        i_proc, n_procs, MPI_Wtime() - time_begin);
    part.GatherSolutions();
    part.WriteSolutions("Step0");
    part.WriteSolutionsOnCellCenters("Step0");
  } else {
    std::printf("Run ReadSolutions(Step%d) on proc[%d/%d] at %f sec\n",
        i_start, i_proc, n_procs, MPI_Wtime() - time_begin);
    part.ReadSolutions("Step" + std::to_string(i_start));
    part.ScatterSolutions();
  }

  auto rk = RungeKutta<kSteps, Part, Riemann>(dt);
  rk.BuildRiemannSolvers(part);

  /* Boundary Conditions */
  auto given_state = [&given_value](const Coord& xyz, double t){
    return given_value;
  };
  if (suffix == "tetra") {
    rk.SetPrescribedBC("3_S_53", given_state);  // Left-Upper
    rk.SetPrescribedBC("3_S_31", given_state);  // Left-Lower
    rk.SetSolidWallBC("3_S_49"); rk.SetSolidWallBC("3_S_71");  // Top
    rk.SetSolidWallBC("3_S_1"); rk.SetSolidWallBC("3_S_2");
    rk.SetSolidWallBC("3_S_3");  // Back
    rk.SetSolidWallBC("3_S_54"); rk.SetSolidWallBC("3_S_76");
    rk.SetSolidWallBC("3_S_32");  // Front
    rk.SetSolidWallBC("3_S_19"); rk.SetSolidWallBC("3_S_23");
    rk.SetSolidWallBC("3_S_63");  // Step
    rk.SetFreeOutletBC("3_S_67");  // Right
  } else {
    assert(suffix == "hexa");
    rk.SetPrescribedBC("4_S_53", given_state);  // Left-Upper
    rk.SetPrescribedBC("4_S_31", given_state);  // Left-Lower
    rk.SetSolidWallBC("4_S_49"); rk.SetSolidWallBC("4_S_71");  // Top
    rk.SetSolidWallBC("4_S_1"); rk.SetSolidWallBC("4_S_2");
    rk.SetSolidWallBC("4_S_3");  // Back
    rk.SetSolidWallBC("4_S_54"); rk.SetSolidWallBC("4_S_76");
    rk.SetSolidWallBC("4_S_32");  // Front
    rk.SetSolidWallBC("4_S_19"); rk.SetSolidWallBC("4_S_23");
    rk.SetSolidWallBC("4_S_63");  // Step
    rk.SetFreeOutletBC("4_S_67");  // Right
  }

  auto wtime_start = MPI_Wtime();
  /* Main Loop */
  for (int i_step = i_start + 1; i_step <= i_stop; ++i_step) {
    double t_curr = t_start + dt * (i_step - i_start - 1);
    rk.Update(&part, t_curr, limiter);

    auto wtime_curr = MPI_Wtime() - wtime_start;
    auto wtime_left = wtime_curr * (i_stop - i_step) / (i_step - i_start);
    std::printf("[Done] Update(Step%d/%d) on proc[%d/%d] at %fs (%fs to go)\n",
        i_step, i_stop, i_proc, n_procs, wtime_curr, wtime_left);

    if (i_step % n_steps_per_frame == 0) {
      std::printf("Run WriteSolutions(Step%d) on proc[%d/%d] at %f sec\n",
          i_step, i_proc, n_procs, MPI_Wtime() - time_begin);
      part.GatherSolutions();
      auto step_name = "Step" + std::to_string(i_step);
      if (i_step == i_stop)
        part.WriteSolutions(step_name);
      part.WriteSolutionsOnCellCenters(step_name);
    }
  }

  std::printf("rank = %d, time = [%f, %f], step = [%d, %d], dt = %f\n",
      i_proc, t_start, t_stop, i_start, i_stop, dt);

  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  MPI_Finalize();
}
