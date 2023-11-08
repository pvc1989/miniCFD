//  Copyright 2022 PEI Weicheng
#include "sourceless.hpp"

/* Set initial conditions. */
auto primitive = Primitive(1.4, 3.0, 0.0, 0.0, 1.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Global &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Global& xyz, double t){ return given_value; };

void MyBC(const std::string &suffix, Solver *solver) {
  if (suffix == "tetra") {
    solver->SetSupersonicInlet("3_S_53", given_state);  // Left-Upper
    solver->SetSupersonicInlet("3_S_31", given_state);  // Left-Lower
    solver->SetSolidWall("3_S_49"); solver->SetSolidWall("3_S_71");  // Top
    solver->SetSolidWall("3_S_1"); solver->SetSolidWall("3_S_2");
    solver->SetSolidWall("3_S_3");  // Back
    solver->SetSolidWall("3_S_54"); solver->SetSolidWall("3_S_76");
    solver->SetSolidWall("3_S_32");  // Front
    solver->SetSolidWall("3_S_19"); solver->SetSolidWall("3_S_23");
    solver->SetSolidWall("3_S_63");  // Step
    solver->SetSupersonicOutlet("3_S_67");  // Right
  } else {
    assert(suffix == "hexa");
    solver->SetSupersonicInlet("4_S_53", given_state);  // Left-Upper
    solver->SetSupersonicInlet("4_S_31", given_state);  // Left-Lower
    solver->SetSolidWall("4_S_49"); solver->SetSolidWall("4_S_71");  // Top
    solver->SetSolidWall("4_S_1"); solver->SetSolidWall("4_S_2");
    solver->SetSolidWall("4_S_3");  // Back
    solver->SetSolidWall("4_S_54"); solver->SetSolidWall("4_S_76");
    solver->SetSolidWall("4_S_32");  // Front
    solver->SetSolidWall("4_S_19"); solver->SetSolidWall("4_S_23");
    solver->SetSolidWall("4_S_63");  // Step
    solver->SetSupersonicOutlet("4_S_67");  // Right
  }
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
