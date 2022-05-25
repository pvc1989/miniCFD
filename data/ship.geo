// Gmsh project created on Sat Mar 26 11:03:31 2022
SetFactory("OpenCASCADE");
ft = 0.3048;
// deck
deck_length = 90 * ft;
deck_width = 45 * ft;
deck_height = 15 * ft;
Box(1) = {0, 0, 0, deck_width, deck_length, deck_height};
// body
Box(2) = {0, 90 * ft, 0, 45 * ft, 230 * ft, 35 * ft};
// tower
Box(3) = {17.5 * ft, 140 * ft, 35 * ft, 10 * ft, 20 * ft, 20 * ft};
// nose
Point(25) = {22.5 * ft, 455 * ft, 0, 1.0 * ft};
Line(37) = {25, 12};
Line(38) = {16, 25};
Curve Loop(19) = {37, 23, 38};
Plane Surface(19) = {19};
Extrude {0, 0, 15 * ft} { Surface{19}; }
// wake
Box(5) = {-52.5 * ft, -100 * ft, 0, 150 * ft, 300 * ft, 85 * ft};
BooleanDifference(6) = { Volume{5}; Delete; }{ Volume{1, 2, 3, 4}; };
// bounding box
Box(7) = {-280 * ft, -900 * ft, 0, 600 * ft, 2000 * ft, 500 * ft};
BooleanDifference(8) = { Volume{7}; Delete; }{ Volume{1, 2, 3, 4, 6}; };
// mesh size
lc = 5 * ft;
Characteristic Length{ PointsOf{ Surface{40};} } = lc * 2;  // sea
Characteristic Length{ PointsOf{ Volume{8};} } = lc * 8;  // far-field
Characteristic Length{ PointsOf{ Volume{1};} } = lc * 1;  // deck
Characteristic Length{ PointsOf{ Volume{2};} } = lc * 2;  // body
Characteristic Length{ PointsOf{ Volume{3};} } = lc * 1;  // tower
Characteristic Length{ PointsOf{ Volume{4};} } = lc * 2;  // nose
Characteristic Length{ PointsOf{ Volume{6};} } = lc * 1;  // wake

Recursive Delete { Volume{1, 2, 3, 4}; }

Mesh 3;
