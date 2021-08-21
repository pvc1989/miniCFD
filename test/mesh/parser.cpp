//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include "mini/mesh/cgns/parser.hpp"

#include "gtest/gtest.h"
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

TEST_F(TestParser, Print) {
  auto cgns_file = current_binary_dir_ + "/ugrid_2d_shuffled.cgns";
  auto prefix = current_binary_dir_ + "/ugrid_2d_part_";
  std::cout << "my_rank = " << rank << std::endl;
  auto parser = Parser<cgsize_t, double>(cgns_file, prefix, rank);
}
int TestParser::rank;

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int comm_size, comm_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  cgp_mpi_comm(MPI_COMM_WORLD);

  std::cout << "comm_rank = " << comm_rank << ", ";
  mini::mesh::cgns::TestParser::rank = comm_rank;

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

  MPI_Finalize();
}
