// circle.geo
SetFactory("OpenCASCADE");
R = 1.0;
Point(1) = {0, 0, 0, 0.1};
Circle(2) = {1, {0,0,0}, R};
Line Loop(3) = {2};
Plane Surface(4) = {3};
Physical Surface("Domain") = {4};
Physical Curve("Boundary") = {2};