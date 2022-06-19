SetFactory("OpenCASCADE");
Box(1) = {-2, -1, -2, 6, 2, 4};

Physical Surface("Left") = { 1 };
Physical Surface("Right") = { 2 };
Physical Surface("Front") = { 3 };
Physical Surface("Back") = { 4 };
Physical Surface("Bottom") = { 5 };
Physical Surface("Top") = { 6 };
Physical Volume("Fluid") = { 1 };

LC = 0.4;
Characteristic Length{ PointsOf{ Volume{1};} } = LC;
Mesh 2;
