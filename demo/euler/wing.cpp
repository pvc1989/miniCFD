//  Copyright 2022 PEI Weicheng
#include "rotorcraft.hpp"

/* Set initial conditions. */
auto primitive = Primitive(1.4, 0.3, 0.0, 0.0, 1.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Coord &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Coord& xyz, double t){
  return given_value;
};

void MyBC(const std::string &suffix, Solver *solver) {
  // Left
  solver->SetSubsonicInlet("3_S_10", given_state);
  // Right
  solver->SetSubsonicOutlet("3_S_15", given_state);
  // Top & Bottom
  solver->SetSolidWall("3_S_12");
  solver->SetSolidWall("3_S_14");
  // Front & Back
  solver->SetSolidWall("3_S_7");
  solver->SetSolidWall("3_S_9");
  solver->SetSolidWall("3_S_11");
  solver->SetSolidWall("3_S_13");
}

int main(int argc, char* argv[]) {
  auto rotor = Source();
  rotor.SetRevolutionsPerSecond(0.0);
  rotor.SetOrigin(0.0, -1.2, 0.0);
  auto frame = Frame();
  frame.RotateY(+10.0/* deg */);
  rotor.SetFrame(frame);
  // build a blade
  std::vector<double> y_values{0.0, 1.5}, chords{0.3, 0.1},
      twists{+10.0, +10.0};
  auto airfoils = std::vector<mini::aircraft::airfoil::SC1095<double>>(3);
  auto blade = Blade();
  blade.InstallSection(y_values[0], chords[0], twists[0], airfoils[0]);
  blade.InstallSection(y_values[1], chords[1], twists[1], airfoils[1]);
  double root{0.2};  // y_root = -1.0, y_tip = 0.5
  rotor.InstallBlade(root, blade);
  rotor.SetInitialAzimuth(0.0);

  return Main(argc, argv, MyIC, MyBC, rotor);
}
