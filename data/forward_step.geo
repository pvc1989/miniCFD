
LC = 0.05;  // the characteristic length, i.e. the average length of cell edges
LX = 3.0;     // the length along x-axis
LY = 1.0;     // the length along y-axis
LZ = 4*LC;  // the length along z-axis
LW = 0.6;   // the width of the bottom before the the step
LH = 0.2;   // the height of the step

Point(1) = { 0, 0, 0, LC };
Point(2) = { LW, 0, 0, LC };
Point(3) = { 0, LH, 0, LC };
Point(4) = { LW, LH, 0, LC };
Point(5) = { LX, LH, 0, LC };
Point(6) = { 0, LY, 0, LC };
Point(7) = { LW, LY, 0, LC };
Point(8) = { LX, LY, 0, LC };

Line(1) = { 1, 2 };
Line(2) = { 3, 4 };
Line(3) = { 4, 5 };
Line(4) = { 6, 7 };
Line(5) = { 7, 8 };
Line(6) = { 1, 3 };
Line(7) = { 3, 6 };
Line(8) = { 2, 4 };
Line(9) = { 4, 7 };
Line(10) = { 5, 8 };

Curve Loop(1) = { 1, 8, -2, -6 };
Curve Loop(2) = { 2, 9, -4, -7 };
Curve Loop(3) = { 3, 10, -5, -9 };
Plane Surface(1) = { 1 };
Plane Surface(2) = { 2 };
Plane Surface(3) = { 3 };

Transfinite Curve{ 1, 2, 4 } = LW/LC + 1.01;
Transfinite Curve{ 3, 5 } = (LX-LW)/LC + 1.01;
Transfinite Curve{ 6, 8 } = LH/LC + 1.01;
Transfinite Curve{ 7, 9, 10 } = (LY-LH)/LC + 1.01;
Transfinite Surface{ 1, 2, 3 };

Recombine Surface{ 1, 2, 3 };
out[] = Extrude{ 0, 0, LZ }{
  Surface{ 1, 2, 3 }; Layers{ LZ/LC+0.001 }; Recombine;
};

Physical Surface("Step") = { 19, 23, 63 };
Physical Surface("Right") = { 67 };
Physical Surface("Top") = { 49, 71 };
Physical Surface("Left") = { 31, 53 };
Physical Surface("Front") = { 32, 54, 76 };
Physical Surface("Back") = { 1, 2, 3 };
Physical Volume("Fluid") = { 1, 2, 3 };
Mesh 3;
