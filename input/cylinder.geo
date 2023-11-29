SetFactory("OpenCASCADE");

Field[1] = MathEval;
Field[1].F = "0.2 * sqrt(x * x + y * y)";
Background Field = 1;

LX = 20;
LY = 20;
Rectangle(1) = {-LX, -LY, 0, LX+LX, LY+LY};
Disk(2) = {0, 0, 0, 1};
BooleanDifference(3) = { Surface{1}; Delete; }{ Surface{2}; Delete; };
Recombine Surface {3};
out[] = Extrude{ 0, 0, 10 }{
  Surface{ 3 }; Layers{ 1 }; Recombine;
};

Physical Surface("Left") = { 5 };
Physical Surface("Right") = { 6 };
Physical Surface("Front") = { 9 };
Physical Surface("Back") = { 3 };
Physical Surface("Top") = { 7 };
Physical Surface("Bottom") = { 4 };
Physical Surface("Cylinder") = { 8 };
Physical Volume("Fluid") = { 1 };

Mesh 3;
