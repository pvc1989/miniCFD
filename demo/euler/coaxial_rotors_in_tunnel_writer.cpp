//  Copyright 2022 PEI Weicheng
#include <vector>

#include "rotorcraft.hpp"

#include "mini/dataset/shuffler.hpp"

int main(int argc, char* argv[]) {
  // Parameters set below must be exactly the same with the solver!
  auto source = Source();
  auto rotor = Rotor();
  auto kOmega = 30.0;
  // Set parameters for the 1st rotor:
  rotor.SetRevolutionsPerSecond(+kOmega);  // right-hand rotation
  rotor.SetOrigin(0.0, 0.0, 0.0);
  auto frame = Frame();
  frame.RotateY(-90.0/* deg */);
  rotor.SetFrame(frame);
  // build a blade
  std::vector<double> y_values{0.0, 1.6}, chords{0.1, 0.1},
      twists{+10.0, +10.0};
  auto airfoils = std::vector<mini::aircraft::airfoil::SC1095<double>>(2);
  auto blade = Blade();
  blade.InstallSection(y_values[0], chords[0], twists[0], airfoils[0]);
  blade.InstallSection(y_values[1], chords[1], twists[1], airfoils[1]);
  double root{0.1};
  rotor.InstallBlade(root, blade);
  rotor.InstallBlade(root, blade);
  rotor.InstallBlade(root, blade);
  rotor.SetInitialAzimuth(0.0);
  source.InstallRotor(rotor);
  // Set parameters for the 2nd rotor:
  rotor.SetRevolutionsPerSecond(-kOmega);  // left-hand rotation
  rotor.SetOrigin(4.0, 0.0, 0.0);
  source.InstallRotor(rotor);
  // Parameters set above must be exactly the same with the solver!

  // int Main(int argc, char* argv[], IC ic, BC bc, Source source)
{
  MPI_Init(NULL, NULL);
  int n_cores, i_core;
  MPI_Comm_size(MPI_COMM_WORLD, &n_cores);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  if (argc < 9) {
    if (i_core == 0) {
      std::cout << "usage:\n"
          << "  mpirun -n <n_cores> " << argv[0] << " <cgns_file> <tetra>"
          << " <t_start> <t_stop> <n_steps_per_frame> <n_frames>"
          << " <i_frame_start> <n_parts_prev>\n";
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
  int i_frame_start = std::atoi(argv[7]);
  int n_parts_prev = std::atoi(argv[8]);

  auto case_name = std::string("coaxial_rotors_in_tunnel");
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

  /* Main Loop */
  auto wtime_start = MPI_Wtime();
  for (int i_frame = i_frame_start; i_frame <= i_frame_start + n_frames;
      ++i_frame) {

    std::string soln_name = (n_parts_prev != n_cores)
        ? "shuffled" : "Frame" + std::to_string(i_frame);
    part.ReadSolutions(soln_name);
    part.ScatterSolutions();
    if (i_core == 0) {
      std::printf("[Done] ReadSolutions(Frame%d) on %d cores at %f sec\n",
          i_frame, n_cores, MPI_Wtime() - time_begin);
    }

    int i_step = 1 + n_steps_per_frame * (i_frame - i_frame_start);
    double t_curr = t_start + dt * (i_step - 1);

    auto frame_name = "Frame" + std::to_string(i_frame);
    WriteForces(part, &source, t_curr, frame_name, i_core);
    if (i_core == 0) {
      std::printf("[Done] WriteForces(Frame%d) on %d cores at %f sec\n",
          i_frame, n_cores, MPI_Wtime() - wtime_start);
    }
  }

  MPI_Finalize();
  return 0;
}
}
