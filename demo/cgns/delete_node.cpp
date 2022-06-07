// Copyright 2022 PEI Weicheng
#include <cstdio>
#include <cstdlib>

#include "cgnslib.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::printf("usage:\n  ");
    std::printf("%s file_name node_name\n", argv[0]);
    exit(0);
  }
  int ier, i_file, i_base = 1, i_zone = 1;
  if (cg_open(argv[1], CG_MODE_MODIFY, &i_file))
    cg_error_exit();

  if (cg_goto(i_file, i_base, "Zone_t", i_zone, "end"))
    cg_error_exit();
  cg_delete_node(argv[2]);

  return cg_close(i_file);
}
