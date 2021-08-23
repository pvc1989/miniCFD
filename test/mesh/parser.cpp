//  Copyright 2021 PEI Weicheng and JIANG Yuyan
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
  auto cgns_file = current_binary_dir_ + "/ugrid_2d_shuffled.cgns";
  auto prefix = current_binary_dir_ + "/ugrid_2d_part_";
  std::cout << "my_rank = " << rank << std::endl;
  auto parser = Parser<cgsize_t, double>(cgns_file, prefix, rank);
}

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  // std::system("./shuffler");

  MPI_Init(NULL, NULL);
  int comm_size, comm_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  cgp_mpi_comm(MPI_COMM_WORLD);

  auto current_binary_dir =
      std::string(PROJECT_BINARY_DIR) + std::string("/test/mesh");
  auto cgns_file = current_binary_dir + "/hexa_new.cgns";
  auto prefix = current_binary_dir + "/hexa_part_";
  auto parser = mini::mesh::cgns::Parser<cgsize_t, double>(
      cgns_file, prefix, comm_rank);

  MPI_Finalize();
}
