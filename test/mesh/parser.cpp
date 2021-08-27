//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdlib>

#include "gtest/gtest.h"
#include "mpi.h"

#include "mini/mesh/cgns/parser.hpp"
#include "mini/data/path.hpp"  // defines PROJECT_BINARY_DIR

namespace mini {
namespace mesh {
namespace cgns {

class TestParser : public ::testing::Test {
 protected:
  std::string const current_binary_dir_{
      std::string(PROJECT_BINARY_DIR) + std::string("/test/mesh")};
 public:
  static int rank;
};
int TestParser::rank;

TEST_F(TestParser, Print) {
}

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int comm_size, comm_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  cgp_mpi_comm(MPI_COMM_WORLD);

  auto time_begin = MPI_Wtime();
  if (comm_rank == 0) {
    std::printf("Run ./shuffler on proc[%d/%d] at %f sec\n",
        comm_rank, comm_size, MPI_Wtime() - time_begin);
    std::system("./shuffler");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::printf("Run Parser() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  auto current_binary_dir =
      std::string(PROJECT_BINARY_DIR) + std::string("/test/mesh");
  auto cgns_file = current_binary_dir + "/hexa_new.cgns";
  auto prefix = current_binary_dir + "/hexa_part_";
  auto parser = mini::mesh::cgns::Parser<cgsize_t, double>(
      cgns_file, prefix, comm_rank);
  std::printf("Run Project() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  parser.Project([](auto const& xyz){
    auto r = std::hypot(xyz[0] - 2, xyz[1] - 0.5);
    mini::algebra::Matrix<double, 2, 1> col;
    col[0] = r;
    col[1] = 1 - r + (r >= 1);
    return col;
  });
  std::printf("Run Write() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  parser.WriteSolutions();
  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  MPI_Finalize();
}
