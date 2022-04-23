//  Copyright 2022 PEI Weicheng
#include <algorithm>
#include <string>
#include <vector>

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/stepper/explicit.hpp"
#include "mini/aircraft/source.hpp"
#include "mini/dataset/shuffler.hpp"

#include "mpi.h"
#include "pcgnslib.h"

/* Define the Euler system. */
constexpr int kDimensions = 3;
using Gas = mini::riemann::euler::IdealGas<double, 1, 4>;
using Unrotated = mini::riemann::euler::Exact<Gas, kDimensions>;
using Riemann = mini::riemann::rotated::Euler<Unrotated>;
using Primitive = typename Riemann::Primitive;
using Conservative = typename Riemann::Conservative;

constexpr int kDegrees = 2;
using Part = mini::mesh::cgns::Part<cgsize_t, kDegrees, Riemann>;
using Cell = typename Part::Cell;
using Face = typename Part::Face;
using Coord = typename Cell::Coord;
using Value = typename Cell::Value;
using Coeff = typename Cell::Coeff;

// using Limiter = mini::polynomial::LazyWeno<Cell>;
using Limiter = mini::polynomial::EigenWeno<Cell>;

using Source = mini::aircraft::RotorSource<Part, double>;
using Rotor = mini::aircraft::Rotor<double>;
using Blade = typename Rotor::Blade;
using Frame = typename Blade::Frame;
using Airfoil = typename Blade::Airfoil;

/* Choose the time-stepping scheme. */
constexpr int kSteps = std::min(3, kDegrees + 1);
using Solver = RungeKutta<kSteps, Part, Limiter, Source>;

using IC = Value(*)(const Coord &);
using BC = void(*)(const std::string &, Solver *);

/* Set initial conditions. */
auto primitive = Primitive(1.4, 0.5, 0.0, 0.0, 1.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Coord &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Coord& xyz, double t){
  return given_value;
};

void MyBC(const std::string &suffix, Solver *solver) {
  solver->SetSubsonicInlet("3_S_1", given_state);
  solver->SetSubsonicOutlet("3_S_2", given_state);
  solver->SetSubsonicOutlet("3_S_3", given_state);
  solver->SetSubsonicOutlet("3_S_4", given_state);
  solver->SetSubsonicOutlet("3_S_5", given_state);
  solver->SetSubsonicOutlet("3_S_6", given_state);
}

int Main(int argc, char* argv[], IC ic, BC bc) {
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

  auto case_name = std::string(argv[0]);
  auto pos = case_name.find_last_of('/');
  if (pos != std::string::npos) {
    case_name = case_name.substr(pos+1);
  }
  case_name.push_back('_');
  case_name += suffix;

  auto time_begin = MPI_Wtime();

  /* Partition the mesh. */
  if (i_core == 0 && n_parts_prev != n_cores) {
    using Shuffler = mini::mesh::Shuffler<idx_t, double>;
    Shuffler::PartitionAndShuffle(case_name, old_file_name, n_cores);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (i_core == 0) {
    std::printf("Create %d `Part`s at %f sec\n",
        n_cores, MPI_Wtime() - time_begin);
  }
  auto part = Part(case_name, i_core);
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
      std::printf("[Done] `Project()` on %d cores at %f sec\n",
          n_cores, MPI_Wtime() - time_begin);
    }

    part.Reconstruct(limiter);
    if (suffix == "tetra") {
      part.Reconstruct(limiter);
    }
    if (i_core == 0) {
      std::printf("[Done] `Reconstruct()` on %d cores at %f sec\n",
          n_cores, MPI_Wtime() - time_begin);
    }

    part.GatherSolutions();
    part.WriteSolutions("Frame0");
    part.WriteSolutionsOnCellCenters("Frame0");
    if (i_core == 0) {
      std::printf("[Done] `WriteSolutions(Frame0)` on %d cores at %f sec\n",
          n_cores, MPI_Wtime() - time_begin);
    }
  } else {
    part.ReadSolutions("Frame" + std::to_string(i_frame));
    part.ScatterSolutions();
    if (i_core == 0) {
      std::printf("[Done] `ReadSolutions(Frame%d)` on %d cores at %f sec\n",
          i_frame, n_cores, MPI_Wtime() - time_begin);
    }
  }

  auto rotor = Source();
  rotor.SetRevolutionsPerSecond(0.0);
  rotor.SetOrigin(0.0, -1.2, 0.0);
  auto frame = Frame();
  frame.RotateY(+10.0/* deg */);
  rotor.SetFrame(frame);
  // build a blade
  std::vector<double> y_values{0.0, 1.1, 2.2}, chords{0.1, 0.3, 0.1},
      twists{-5.0, -5.0, -5.0};
  auto airfoils = std::vector<mini::aircraft::airfoil::Simple<double>>();
  auto c_lift = mini::geometry::deg2rad(2 * mini::geometry::pi());
  airfoils.emplace_back(c_lift, 0.0);
  airfoils.emplace_back(c_lift, 0.0);
  airfoils.emplace_back(c_lift, 0.0);
  auto blade = Blade();
  blade.InstallSection(y_values[0], chords[0], twists[0], airfoils[0]);
  blade.InstallSection(y_values[1], chords[1], twists[1], airfoils[1]);
  blade.InstallSection(y_values[2], chords[2], twists[2], airfoils[2]);
  double root{0.1};
  rotor.InstallBlade(root, blade);
  rotor.SetAzimuth(0.0);

  /* Choose the time-stepping scheme. */
  auto rk = Solver(dt, limiter, rotor);

  /* Set boundary conditions. */
  bc(suffix, &rk);

  /* Main Loop */
  auto wtime_start = MPI_Wtime();
  for (int i_step = 1; i_step <= n_steps; ++i_step) {
    double t_curr = t_start + dt * (i_step - 1);
    rk.Update(&part, t_curr);

    auto wtime_curr = MPI_Wtime() - wtime_start;
    auto wtime_left = wtime_curr * (n_steps - i_step) / (i_step);
    if (i_core == 0) {
      std::printf("[Done] `Update(Step%d/%d)` on %d cores at %fs (%fs to go)\n",
          i_step, n_steps, n_cores, wtime_curr, wtime_left);
    }

    if (i_step % n_steps_per_frame == 0) {
      ++i_frame;
      auto frame_name = "Frame" + std::to_string(i_frame);
      part.GatherSolutions();
      part.WriteSolutions(frame_name);
      part.WriteSolutionsOnCellCenters(frame_name);
      if (i_core == 0) {
        std::printf("[Done] `WriteSolutions(Frame%d)` on %d cores at %f sec\n",
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

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
