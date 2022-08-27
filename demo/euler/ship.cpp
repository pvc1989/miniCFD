//  Copyright 2022 PEI Weicheng
#include "sourceless.hpp"

/* Set initial conditions. */
auto primitive = Primitive(1.29, 5.0, 0.0, 0.0, 101325.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Coord &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Coord& xyz, double t){
  return given_value;
};

void MyBC(const std::string &suffix, Solver *solver) {
  // bounding box
  solver->SetSmartBoundary("3_S_37"/* Front */, given_state);
  solver->SetSmartBoundary("3_S_36"/* Left */, given_state);
  solver->SetSmartBoundary("3_S_41"/* Right */, given_state);
  solver->SetSmartBoundary("3_S_38"/* Top */, given_state);
  solver->SetSmartBoundary("3_S_39"/* Back */, given_state);
  // bottom
  solver->SetSolidWall("3_S_27"/* fine */);
  solver->SetSolidWall("3_S_40"/* coarse */);
  // tower
  solver->SetSolidWall("3_S_13");
  solver->SetSolidWall("3_S_14");
  solver->SetSolidWall("3_S_15");
  solver->SetSolidWall("3_S_16");
  solver->SetSolidWall("3_S_18");
  // nose
  solver->SetSolidWall("3_S_21");
  solver->SetSolidWall("3_S_22");
  solver->SetSolidWall("3_S_23");
  // body
  solver->SetSolidWall("3_S_28");
  solver->SetSolidWall("3_S_29");
  solver->SetSolidWall("3_S_30");
  solver->SetSolidWall("3_S_35");
  solver->SetSolidWall("3_S_42");
  solver->SetSolidWall("3_S_43");
  solver->SetSolidWall("3_S_44");
  solver->SetSolidWall("3_S_45");
  // deck
  solver->SetSolidWall("3_S_2");
  solver->SetSolidWall("3_S_6");
  solver->SetSolidWall("3_S_33");
  solver->SetSolidWall("3_S_34");
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
