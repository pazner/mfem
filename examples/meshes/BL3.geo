Parametric Surface(100) = "u" "v" "sqrt(9 - u^2 - v^2)";

// make it the active coordinate system
Coordinates Surface 100;

Point(1) = {-1.7, -0.3, 0, .1};
Point(2) = {-1.2, -0, 0, .1};
Point(3) = {-0.5, -0.3, 0, .1};
Point(4) = {-0.4, -0.7, 0, .1};
Point(5) = {-0.8, -0.8, 0, .1};
Point(6) = {-1.2, -0.5, 0, .1};
Point(7) = {-1.8, -0.5, 0, .1};
Point(8) = {-1.1, 0.6, 0, .1};
Point(9) = {-0.5, 0.7, 0, .1};
Point(10) = {-0.1, 0.2, 0, .1};
Point(11) = {-0.3, -0.1, 0, .1};
Point(12) = {-0.7, 0.2, 0, .1};
Point(13) = {-1.1, 0.4, 0, 1.0};
Point(14) = {-1.5, 0.5, 0, .1};
Point(15) = {-2.5, 1.5, 0, .1};
Point(16) = {-2.5, -1.5, 0, .1};
Point(17) = {1.5, -1.5, 0, .1};
Point(18) = {1.5, 1.5, 0, .1};
Line(1) = {15, 16};
Line(2) = {16, 17};
Line(3) = {17, 18};
Line(4) = {18, 15};
Spline(5) = {14, 8, 9, 10, 11, 12, 13, 14};
Spline(6) = {2, 3, 4, 5, 6, 7, 1, 2};
Line Loop(7) = {1, 2, 3, 4};
Line Loop(8) = {5};
Line Loop(9) = {6};
Plane Surface(10) = {7, 8, 9};

Physical Surface(11) = {10};

Mesh.MeshSizeFromPoints = 0;

Field[1] = Distance;
Field[1].CurvesList = {5, 6};
Field[1].Sampling = 150;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.08;
Field[2].SizeMax = 0.5;
Field[2].DistMin = 0.1;
Field[2].DistMax = 0.5;

Background Field = 2;
