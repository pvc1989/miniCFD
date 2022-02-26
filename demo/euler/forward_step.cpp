#include "main.hpp"

/* Set initial conditions. */
auto primitive = Primitive(1.4, 3.0, 0.0, 0.0, 1.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Coord &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Coord& xyz, double t){ return given_value; };

void MyBC(const std::string &suffix, Solver *solver) {
  if (suffix == "tetra") {
    solver->SetPrescribedBC("3_S_53", given_state);  // Left-Upper
    solver->SetPrescribedBC("3_S_31", given_state);  // Left-Lower
    solver->SetSolidWallBC("3_S_49"); solver->SetSolidWallBC("3_S_71");  // Top
    solver->SetSolidWallBC("3_S_1"); solver->SetSolidWallBC("3_S_2");
    solver->SetSolidWallBC("3_S_3");  // Back
    solver->SetSolidWallBC("3_S_54"); solver->SetSolidWallBC("3_S_76");
    solver->SetSolidWallBC("3_S_32");  // Front
    solver->SetSolidWallBC("3_S_19"); solver->SetSolidWallBC("3_S_23");
    solver->SetSolidWallBC("3_S_63");  // Step
    solver->SetFreeOutletBC("3_S_67");  // Right
  } else {
    assert(suffix == "hexa");
    solver->SetPrescribedBC("4_S_53", given_state);  // Left-Upper
    solver->SetPrescribedBC("4_S_31", given_state);  // Left-Lower
    solver->SetSolidWallBC("4_S_49"); solver->SetSolidWallBC("4_S_71");  // Top
    solver->SetSolidWallBC("4_S_1"); solver->SetSolidWallBC("4_S_2");
    solver->SetSolidWallBC("4_S_3");  // Back
    solver->SetSolidWallBC("4_S_54"); solver->SetSolidWallBC("4_S_76");
    solver->SetSolidWallBC("4_S_32");  // Front
    solver->SetSolidWallBC("4_S_19"); solver->SetSolidWallBC("4_S_23");
    solver->SetSolidWallBC("4_S_63");  // Step
    solver->SetFreeOutletBC("4_S_67");  // Right
  }
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
