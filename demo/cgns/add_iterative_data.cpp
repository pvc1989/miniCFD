// Copyright 2022 PEI Weicheng
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "cgnslib.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::printf("usage:\n  ");
    std::printf("%s file_name frame_min frame_max\n", argv[0]);
    exit(0);
  }
  int ier, i_file, i_base = 1, i_zone = 1, frame_min, frame_max, frame_cnt;
  if (cg_open(argv[1], CG_MODE_MODIFY, &i_file))
    cg_error_exit();
  frame_min = std::atoi(argv[2]);
  frame_max = std::atoi(argv[3]);
  frame_cnt = frame_max - frame_min + 1;

  if (cg_simulation_type_write(i_file, i_base, CGNS_ENUMV(TimeAccurate)))
    cg_error_exit();

  cg_delete_node("TimeIterValues");
  if (cg_biter_write(i_file, i_base, "TimeIterValues", frame_cnt))
    cg_error_exit();

  if (cg_goto(i_file, i_base, "Zone_t", i_zone, "end"))
    cg_error_exit();
  cg_delete_node("ZoneIterativeData");
  if (cg_ziter_write(i_file, i_base, i_zone, "ZoneIterativeData"))
    cg_error_exit();
  if (cg_goto(i_file, i_base, "Zone_t", i_zone,
      "ZoneIterativeData_t", 1, "end"))
    cg_error_exit();

  constexpr int kNameLength = 32;
  /* need an extra byte for the terminating '\0' */
  auto names = std::string(kNameLength * frame_cnt + 1, '\0');
  char* head = names.data();
  for (int frame = frame_min; frame <= frame_max; ++frame) {
    std::snprintf(head, kNameLength + 1, "Frame%d", frame);
    std::printf("%s\n", head);
    head += kNameLength;
  }
  cgsize_t n_dims[2] = {kNameLength, frame_cnt};
  if (cg_array_write("FlowSolutionPointers", CGNS_ENUMV(Character), 2, n_dims,
      names.c_str()/* the head of a char[frame_cnt][kNameLength] 2d array */))
    cg_error_exit();
  return cg_close(i_file);
}
