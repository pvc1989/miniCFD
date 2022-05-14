//  Copyright 2022 PEI Weicheng
#include "rotor_source.hpp"

/* Set initial conditions. */
auto primitive = Primitive(1.4, 0.02, 0.0, 0.0, 1.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Coord &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Coord& xyz, double t){
  return given_value;
};

void MyBC(const std::string &suffix, Solver *solver) {
  // Left & Top
  solver->SetSubsonicInlet("3_S_10", given_state);
  solver->SetSubsonicInlet("3_S_12", given_state);
  // Front & Back
  solver->SetSubsonicOutlet("3_S_11", given_state);
  solver->SetSubsonicOutlet("3_S_13", given_state);
  // Bottom & Right
  solver->SetSubsonicOutlet("3_S_14", given_state);
  solver->SetSubsonicOutlet("3_S_15", given_state);
}

int main(int argc, char* argv[]) {
  auto rotor = Source();
  rotor.SetRevolutionsPerSecond(10.0);
  rotor.SetOrigin(0.0, 0.0, 0.5);
  auto frame = Frame();
  frame.RotateY(+0.0/* deg */);
  rotor.SetFrame(frame);
  // build a blade
  std::vector<double> y_values{0.0, 0.9}, chords{0.1, 0.1},
      twists{+5.0, +0.0};
  auto airfoils = std::vector<mini::aircraft::airfoil::SC1095<double>>(2);
  auto blade = Blade();
  blade.InstallSection(y_values[0], chords[0], twists[0], airfoils[0]);
  blade.InstallSection(y_values[1], chords[1], twists[1], airfoils[1]);
  double root{0.1};
  rotor.InstallBlade(root, blade);
  rotor.InstallBlade(root, blade);
  rotor.SetAzimuth(0.0);

  return Main(argc, argv, MyIC, MyBC, rotor);
}
