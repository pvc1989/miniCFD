//  Copyright 2022 PEI Weicheng
#include "rotorcraft.hpp"

/* Set initial conditions. */
auto primitive = Primitive(1.29, 0.0, 0.0, 0.0, 101325.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Coord &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Coord& xyz, double t){
  return given_value;
};

void MyBC(const std::string &suffix, Solver *solver) {
  solver->SetSubsonicInlet("3_S_13"/* Left */, given_state);
  solver->SetSolidWall("3_S_14"/* Bottom */);
  solver->SetSolidWall("3_S_15"/* Front */);
  solver->SetSolidWall("3_S_16"/* Top */);
  solver->SetSolidWall("3_S_17"/* Back */);
  solver->SetSubsonicOutlet("3_S_18"/* Right */, given_state);
}

int main(int argc, char* argv[]) {
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
  std::vector<double> y_values{0.0, 0.9}, chords{0.1, 0.1},
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
  rotor.SetOrigin(0.2, 0.0, 0.0);
  source.InstallRotor(rotor);

  return Main(argc, argv, MyIC, MyBC, source);
}
