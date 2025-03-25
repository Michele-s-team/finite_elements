'''
Ths code generates a 2d mesh given by a square, where the mesh is uniform across the square

run with
clear; clear; python3 generate_mesh.py [resolution]
example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py 0.1 $SOLUTION_PATH
'''

import gmsh
import pygmsh
import argparse
import numpy as np

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
parser.add_argument("output_directory")
args = parser.parse_args()

# mesh resolution
resolution = (float)(args.resolution)
mesh_file = args.output_directory + "/mesh.msh"

# mesh parameters
# CHANGE PARAMETERS HERE
L = 2.2
h = 0.41
# CHANGE PARAMETERS HERE

print("L = ", L)
print("h = ", h)

p_1_id = 1
p_2_id = 2
p_3_id = 3
p_4_id = 4
line_12_id = 5
line_23_id = 6
line_34_id = 7
line_41_id = 8
surface_id = 9

geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

# add a 0d object:
p_1 = gmsh.model.geo.addPoint(0, 0, 0)
p_2 = gmsh.model.geo.addPoint(L, 0, 0)
p_3 = gmsh.model.geo.addPoint(L, h, 0)
p_4 = gmsh.model.geo.addPoint(0, h, 0)

line_12 = gmsh.model.geo.addLine(p_1, p_2)
gmsh.model.geo.synchronize()

line_23 = gmsh.model.geo.addLine(p_2, p_3)
gmsh.model.geo.synchronize()

line_34 = gmsh.model.geo.addLine(p_3, p_4)
gmsh.model.geo.synchronize()

line_41 = gmsh.model.geo.addLine(p_4, p_1)
gmsh.model.geo.synchronize()

resolution_2 = resolution / 2

loop = gmsh.model.geo.addCurveLoop([line_12, line_23, line_34, line_41])
gmsh.model.geo.synchronize()

surface = gmsh.model.geo.addPlaneSurface([loop])
gmsh.model.geo.synchronize()

# add intermediate lines horizontally
n_intermediate_lines = (int)(np.floor(L / resolution_2) - 1)
print(f"n_intermediate_lines = {n_intermediate_lines}")

p_t = []
p_b = []
for i in range(n_intermediate_lines):
    p_t.append(gmsh.model.geo.addPoint(resolution_2 * (i + 1), 0 + resolution_2, 0))
    p_b.append(gmsh.model.geo.addPoint(resolution_2 * (i + 1), h - resolution_2, 0))
    gmsh.model.geo.synchronize()

    line_tb = gmsh.model.geo.addLine(p_t[i], p_b[i])
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.embed(1, [line_tb], 2, surface)
    gmsh.model.geo.synchronize()

# add intermediate lines vertically
# n_intermediate_lines = (int)(np.floor(h / resolution_2) - 1)
# print(f"n_intermediate_lines = {n_intermediate_lines}")

# p_l = []
# p_r = []
# for i in [1,n_intermediate_lines-2]:
#     print(f"i = {i}")
p_l = gmsh.model.geo.addPoint(L/4, h/2, 0)
gmsh.model.geo.synchronize()

p_r = gmsh.model.geo.addPoint(3*L/4, h/2,  0)
gmsh.model.geo.synchronize()

print("p_l = ", p_l)
print("p_r = ", p_r)


line_lr = gmsh.model.geo.addLine(p_l, p_r)
gmsh.model.geo.synchronize()

gmsh.model.mesh.embed(1, [line_lr], 2, surface)
gmsh.model.geo.synchronize()



# add 0-dimensional objects
vertices = gmsh.model.getEntities(dim=0)

gmsh.model.addPhysicalGroup(vertices[0][0], [vertices[0][1]], p_1_id)
gmsh.model.setPhysicalName(vertices[0][0], p_1_id, "p_1")

gmsh.model.addPhysicalGroup(vertices[1][0], [vertices[1][1]], p_2_id)
gmsh.model.setPhysicalName(vertices[1][0], p_2_id, "p_2")

gmsh.model.addPhysicalGroup(vertices[2][0], [vertices[2][1]], p_3_id)
gmsh.model.setPhysicalName(vertices[2][0], p_3_id, "p_3")

gmsh.model.addPhysicalGroup(vertices[3][0], [vertices[3][1]], p_4_id)
gmsh.model.setPhysicalName(vertices[3][0], p_4_id, "p_4")

# add 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

gmsh.model.addPhysicalGroup(lines[0][0], [lines[0][1]], line_12_id)
gmsh.model.setPhysicalName(lines[0][0], line_12_id, "line_12")

gmsh.model.addPhysicalGroup(lines[1][0], [lines[1][1]], line_23_id)
gmsh.model.setPhysicalName(lines[1][0], line_23_id, "arc_12")

gmsh.model.addPhysicalGroup(lines[2][0], [lines[2][1]], line_34_id)
gmsh.model.setPhysicalName(lines[2][0], line_34_id, "line_34")

gmsh.model.addPhysicalGroup(lines[3][0], [lines[3][1]], line_41_id)
gmsh.model.setPhysicalName(lines[3][0], line_41_id, "line_41")

# add 2-dimensional objects
surfaces = gmsh.model.getEntities(dim=2)

gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], surface_id)
gmsh.model.setPhysicalName(surfaces[0][0], surface_id, "surface")

# set the resolution
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [surface])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution/2)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", resolution)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", h)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", L)

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

gmsh.model.geo.synchronize()

geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

# write mesh components to file
msh.write_mesh_components(mesh_file, "solution/triangle_mesh.xdmf", "triangle", True)
msh.write_mesh_components(mesh_file, "solution/line_mesh.xdmf", "line", True)
msh.write_mesh_components(mesh_file, "solution/vertex_mesh.xdmf", "vertex", True)

msh.write_mesh_to_csv(mesh_file, 'solution/line_vertices.csv')

model.__exit__()
