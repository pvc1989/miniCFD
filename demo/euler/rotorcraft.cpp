//  Copyright 2022 PEI Weicheng
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/shuffler.hpp"
#include "mini/mesh/vtk.hpp"

#include "rotorcraft.hpp"

void WriteForces(Part const &part, Source *source, double t_curr,
    std::string const &frame_name, int i_core) {
  using Force = Coord;
  std::vector<Force> forces;
  std::vector<Coord> points;
  std::vector<Scalar> weights;
  part.ForEachConstLocalCell(
      [source, t_curr, &forces, &points, &weights](const Cell &cell){
    source->GetForces(cell, t_curr, &forces, &points, &weights);
  });
  auto out = part.GetFileStream(frame_name, false, "csv");
  out << "\"X\",\"Y\",\"Z\",\"ForceX\",\"ForceY\",\"ForceZ\",\"Weight\"\n";
  for (int i = 0, n = weights.size(); i < n; ++i) {
    out << points[i][0] << ',' << points[i][1] << ',' << points[i][2] << ',';
    out << forces[i][0] << ',' << forces[i][1] << ',' << forces[i][2] << ',';
    out << weights[i] << '\n';
  }
}

int Main(int argc, char* argv[], IC ic, BC bc, Source source) {
  MPI_Init(NULL, NULL);
  int n_cores, i_core;
  MPI_Comm_size(MPI_COMM_WORLD, &n_cores);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  if (argc < 7) {
    if (i_core == 0) {
      std::cout << "usage:\n"
          << "  mpirun -n <n_cores> " << argv[0] << " <cgns_file> <output_path>"
          << " <t_start> <t_stop> <n_steps_per_frame> <n_frames>"
          << " [<i_frame_start> [n_parts_prev] [--write_forces]]\n";
    }
    MPI_Finalize();
    exit(0);
  }
  auto old_file_name = std::string(argv[1]);
  auto output_path = std::string(argv[2]);
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
  bool write_forces = false;
  if (argc > 9) {
    write_forces = true;
  }

  auto time_begin = MPI_Wtime();

  /* Partition the mesh. */
  if (i_core == 0 && n_parts_prev != n_cores) {
    using Shuffler = mini::mesh::Shuffler<idx_t, double>;
    Shuffler::PartitionAndShuffle(output_path, old_file_name, n_cores);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (i_core == 0) {
    std::printf("Create %d `Part`s at %f sec\n",
        n_cores, MPI_Wtime() - time_begin);
  }
  auto part = Part(output_path, i_core);
  part.SetFieldNames({"Density", "MomentumX", "MomentumY", "MomentumZ",
      "EnergyStagnationDensity"});

  /* Build a `Limiter` object. */
  auto limiter = Limiter(/* w0 = */0.001, /* eps = */1e-6);

  /* Initialization. */
  if (argc == 7) {
    part.ForEachLocalCell([&](Cell *cell_ptr){
      cell_ptr->Project(ic);
    });
    if (i_core == 0) {
      std::printf("[Done] Project() on %d cores at %f sec\n",
          n_cores, MPI_Wtime() - time_begin);
    }
    part.GatherSolutions();
    part.WriteSolutions("Frame0");
    mini::mesh::vtk::Writer<Part>::WriteSolutions(part, "Frame0");
    if (i_core == 0) {
      std::printf("[Done] WriteSolutions(Frame0) on %d cores at %f sec\n",
          n_cores, MPI_Wtime() - time_begin);
    }
  } else {
    std::string soln_name = (n_parts_prev != n_cores)
        ? "shuffled" : "Frame" + std::to_string(i_frame);
    part.ReadSolutions(soln_name);
    part.ScatterSolutions();
    if (i_core == 0) {
      std::printf("[Done] ReadSolutions(Frame%d) on %d cores at %f sec\n",
          i_frame, n_cores, MPI_Wtime() - time_begin);
    }
  }
  if (write_forces) {
    auto frame_name = "Frame" + std::to_string(i_frame);
    WriteForces(part, &source, t_start, frame_name, i_core);
    if (i_core == 0) {
      std::printf("[Done] WriteForces(Frame%d) on %d cores at %f sec\n",
          i_frame, n_cores, MPI_Wtime() - time_begin);
    }
  }

  /* Choose the time-stepping scheme. */
  auto solver = Solver(dt, limiter, source);

  /* Set boundary conditions. */
  bc("tetra", &solver);

  /* Main Loop */
  auto wtime_start = MPI_Wtime();
  for (int i_step = 1; i_step <= n_steps; ++i_step) {
    double t_curr = t_start + dt * (i_step - 1);
    std::string frame_name;
    if (write_forces) {
      while (i_step % n_steps_per_frame) {
        ++i_step;
      }
      ++i_frame;
      auto frame_name = "Frame" + std::to_string(i_frame);
      part.ReadSolutions(frame_name);
      part.ScatterSolutions();
      if (i_core == 0) {
        std::printf("[Done] ReadSolutions(Frame%d) on %d cores at %f sec\n",
            i_frame, n_cores, MPI_Wtime() - time_begin);
      }
      WriteForces(part, &source, t_curr, frame_name, i_core);
      if (i_core == 0) {
        std::printf("[Done] WriteForces(Frame%d) on %d cores at %f sec\n",
            i_frame, n_cores, MPI_Wtime() - wtime_start);
      }
      continue;
    }
    solver.Update(&part, t_curr);

    auto wtime_curr = MPI_Wtime() - wtime_start;
    auto wtime_total = wtime_curr * n_steps / i_step;
    if (i_core == 0) {
      std::printf("[Done] Update(Step%d/%d) on %d cores at %f / %f sec\n",
          i_step, n_steps, n_cores, wtime_curr, wtime_total);
    }

    if (i_step % n_steps_per_frame == 0) {
      ++i_frame;
      auto frame_name = "Frame" + std::to_string(i_frame);
      part.GatherSolutions();
      part.WriteSolutions(frame_name);
      mini::mesh::vtk::Writer<Part>::WriteSolutions(part, frame_name);
      if (i_core == 0) {
        std::printf("[Done] WriteSolutions(Frame%d) on %d cores at %f sec\n",
            i_frame, n_cores, MPI_Wtime() - wtime_start);
      }
    }
  }

  if (i_core == 0) {
    std::printf("time-range = [%f, %f], frame-range = [%d, %d], dt = %f\n",
        t_start, t_stop, i_frame - n_frames, i_frame, dt);
    std::printf("[Start] MPI_Finalize() on %d cores at %f sec\n",
        n_cores, MPI_Wtime() - time_begin);
  }
  MPI_Finalize();
  return 0;
}
