SetFactory("OpenCASCADE");
Box(1) = {-2, -2, -2, 6, 4, 4};

Physical Surface("Inlet") = { 1 };
Physical Surface("Outlet") = { 2 };
Physical Surface("Wall") = { 3, 4, 5, 6 };
Physical Volume("Fluid") = { 1 };

LC = 0.4;
Characteristic Length{ PointsOf{ Volume{1};} } = LC;
Mesh 3;
