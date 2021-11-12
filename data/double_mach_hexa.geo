/*
 * Generate unstructured hexahedral mesh for "double mach" problem.
 */
LC = 0.05;  // average length of cell edges, a.k.a. characteristic length
LX = 4;     // length along x-axis
LY = 1;     // length along y-axis
LZ = 4*LC;  // length along z-axis
LG = 1./6;  // length of the gap before the wall

Point(1) = { 0., 0., 0., LC };
Point(2) = { LG, 0., 0., LC };
Point(3) = { LX, 0., 0., LC };
Point(4) = { LX, LY, 0., LC };
Point(5) = { 0., LY, 0., LC };

Line(1) = { 1, 2 };
Line(2) = { 2, 3 };
Line(3) = { 3, 4 };
Line(4) = { 4, 5 };
Line(5) = { 5, 1 };

Curve Loop(1) = { 1, 2, 3, 4, 5 };
Plane Surface(1) = { 1 };
Recombine Surface{ 1 };
out[] = Extrude{ 0, 0, LZ }{
  Surface{ 1 }; Layers{ 4 }; Recombine;
};

Physical Surface("Gap") = { 15 };
Physical Surface("Wall") = { 19 };
Physical Surface("Right") = { 23 };
Physical Surface("Top") = { 27 };
Physical Surface("Left") = { 31 };
Physical Surface("Front") = { 32 };
Physical Surface("Back") = { 1 };
Physical Volume("Fluid") = { 1 };

Mesh 3;
