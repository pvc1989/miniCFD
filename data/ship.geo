// Gmsh project created on Sat Mar 26 11:03:31 2022
SetFactory("OpenCASCADE");
// mesh size
cell_large = 50;
cell_small = 10;
// deck
deck_length = 90;
deck_width = 45;
deck_height = 15;
Box(1) = {0, 0, 0, deck_width, deck_length, deck_height};
// body
Box(2) = {0, 90, 0, 45, 230, 35};
// tower
Box(3) = {17.5, 140, 35, 10, 20, 20};
// nose
Point(25) = {22.5, 455, 0, 1.0};
Line(37) = {25, 12};
Line(38) = {16, 25};
Curve Loop(19) = {37, 23, 38};
Plane Surface(19) = {19};
Extrude {0, 0, 15} {
  Surface{19}; 
}
Characteristic Length{ PointsOf{ Volume{1, 2, 3, 4};} } = cell_small;
// bounding box
Box(5) = {-280, -900, 0, 600, 2000, 500};
Characteristic Length{ PointsOf{ Volume{5};} } = cell_large;
// fluid with wake
BooleanDifference(6) = { Volume{5}; Delete; }{ Volume{1, 2, 3, 4}; Delete; };
// wake
Box(7) = {-52.5, -345, 15 + cell_small, 150, 435 - cell_small, 85};
Characteristic Length{ PointsOf{ Volume{7};} } = cell_small;
// fluid without wake
BooleanDifference(8) = { Volume{6}; Delete; }{ Volume{7}; };
Characteristic Length{ PointsOf{ Surface{34};} } = cell_small;
