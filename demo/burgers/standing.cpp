//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/dataset/shuffler.hpp"
#include "mini/riemann/rotated/burgers.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/solver/rkdg.hpp"
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
          << "  mpirun -n <n_cores> " << argv[0] << " <cgns_file> <hexa|tetra>"
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

  std::string case_name = "standing_" + suffix;

  auto time_begin = MPI_Wtime();

  /* Define the Burgers equation. */
  constexpr int kComponents = 1;
  constexpr int kDimensions = 3;
  using Riemann = mini::riemann::rotated::Burgers<double, kDimensions>;
  Riemann::global_coefficient = { 1, 0, 0 };

  /* Partition the mesh. */
  if (i_core == 0 && n_parts_prev != n_cores) {
    using Shuffler = mini::mesh::Shuffler<idx_t, double>;
    Shuffler::PartitionAndShuffle(case_name, old_file_name, n_cores);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int kDegrees = 2;
  using Part = mini::mesh::cgns::Part<cgsize_t, kDegrees, Riemann>;
  using Cell = typename Part::Cell;
  using Face = typename Part::Face;
  using Coord = typename Cell::Coord;
  using Value = typename Cell::Value;
  using Coeff = typename Cell::Coeff;

  if (i_core == 0) {
    std::printf("Create %d `Part`s at %f sec\n",
        n_cores, MPI_Wtime() - time_begin);
  }
  auto part = Part(case_name, i_core);
  part.SetFieldNames({"U"});

  /* Build a `Limiter` object. */
  using Limiter = mini::polynomial::LazyWeno<Cell>;
  auto limiter = Limiter(/* w0 = */0.001, /* eps = */1e-6);

  /* Set initial conditions. */
  auto initial_condition = [&](const Coord& xyz){
    auto x = xyz[0];
    Value val;
    val[0] = x * (x - 2.0) * (x - 4.0);
    return val;
  };

  if (argc == 7) {
    if (i_core == 0) {
      std::printf("[Start] Project() on %d cores at %f sec\n",
          n_cores, MPI_Wtime() - time_begin);
    }
    part.ForEachLocalCell([&](Cell *cell_ptr){
      cell_ptr->Project(initial_condition);
    });

    if (i_core == 0) {
      std::printf("[Start] Reconstruct() on %d cores at %f sec\n",
          n_cores, MPI_Wtime() - time_begin);
    }
    if (kDegrees > 0) {
      part.Reconstruct(limiter);
      if (suffix == "tetra") {
        part.Reconstruct(limiter);
      }
    }

    part.GatherSolutions();
    if (i_core == 0) {
      std::printf("[Start] WriteSolutions(Frame0) on %d cores at %f sec\n",
          n_cores, MPI_Wtime() - time_begin);
    }
    part.WriteSolutions("Frame0");
    part.WriteSolutionsOnCellCenters("Frame0");
  } else {
    if (i_core == 0) {
      std::printf("[Start] ReadSolutions(Frame%d) on %d cores at %f sec\n",
          i_frame, n_cores, MPI_Wtime() - time_begin);
    }
    part.ReadSolutions("Frame" + std::to_string(i_frame));
    part.ScatterSolutions();
  }

  /* Choose the time-stepping scheme. */
  constexpr int kOrders = std::min(3, kDegrees + 1);
  auto solver = RungeKutta<kOrders, Part, Limiter>(dt, limiter);

  /* Set boundary conditions. */
  if (suffix == "tetra") {
    solver.SetSolidWall("3_S_27");  // Top
    solver.SetSolidWall("3_S_31");  // Left
    solver.SetSolidWall("3_S_1");   // Back
    solver.SetSolidWall("3_S_32");  // Front
    solver.SetSolidWall("3_S_19");  // Bottom
    solver.SetSolidWall("3_S_23");  // Right
    solver.SetSolidWall("3_S_15");  // Gap
  } else {
    assert(suffix == "hexa");
    solver.SetSolidWall("4_S_27");  // Top
    solver.SetSolidWall("4_S_31");  // Left
    solver.SetSolidWall("4_S_1");   // Back
    solver.SetSolidWall("4_S_32");  // Front
    solver.SetSolidWall("4_S_19");  // Bottom
    solver.SetSolidWall("4_S_23");  // Right
    solver.SetSolidWall("4_S_15");  // Gap
  }

  /* Main Loop */
  auto wtime_start = MPI_Wtime();
  for (int i_step = 1; i_step <= n_steps; ++i_step) {
    double t_curr = t_start + dt * (i_step - 1);
    solver.Update(&part, t_curr);

    auto wtime_curr = MPI_Wtime() - wtime_start;
    auto wtime_total = wtime_curr * n_steps / i_step;
    if (i_core == 0) {
      std::printf("[Done] Update(Step%d/%d) on %d cores at %f/%f sec\n",
          i_step, n_steps, n_cores, wtime_curr, wtime_total);
    }

    if (i_step % n_steps_per_frame == 0) {
      ++i_frame;
      part.GatherSolutions();
      if (i_core == 0) {
        std::printf("[Start] WriteSolutions(Frame%d) on %d cores at %f sec\n",
            i_frame, n_cores, MPI_Wtime() - wtime_start);
      }
      auto frame_name = "Frame" + std::to_string(i_frame);
      part.WriteSolutions(frame_name);
      part.WriteSolutionsOnCellCenters(frame_name);
    }
  }

  if (i_core == 0) {
    std::printf("time-range = [%f, %f], frame-range = [%d, %d], dt = %f\n",
        t_start, t_stop, i_frame - n_frames, i_frame, dt);
    std::printf("[Start] MPI_Finalize() on %d cores at %f sec\n",
        n_cores, MPI_Wtime() - time_begin);
  }
  MPI_Finalize();
}
