'''
This code generates a 3d mesh given by a ring with multiple radial lines
IT IS NOT GUARANTEED THAT THIS MESH GENERATES A MESH WITH PERFECT RADIAL SYMMETRY

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/ring/ring_with_lines/"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/ring/ring_with_lines/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import meshio
import numpy as np
import gmsh
import warnings
from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')

output_directory = io.add_trailing_slash(rarg.args.output_directory)

mesh_file = output_directory + "mesh.msh"

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'


warnings.filterwarnings("ignore")
gmsh.initialize()

gmsh.model.add("my model")


delta_theta = 2 * np.pi / rpam.parameters["N"]


def Q(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


x_r = np.array([rpam.parameters["r"] + rpam.parameters["resolution"], 0])
x_R = np.array([rpam.parameters["R"] - rpam.parameters["resolution"], 0])

disk_r = gmsh.model.occ.addDisk(rpam.parameters["c_r"][0], rpam.parameters["c_r"][1], rpam.parameters["c_r"][2], rpam.parameters["r"], rpam.parameters["r"])
disk_R = gmsh.model.occ.addDisk(rpam.parameters["c_R"][0], rpam.parameters["c_R"][1], rpam.parameters["c_R"][2], rpam.parameters["R"], rpam.parameters["R"])
# add this every time you add a component to the mesh and every time you make modifications to the mesh
gmsh.model.occ.synchronize()

ring = gmsh.model.occ.cut([(2, disk_R)], [(2, disk_r)])
gmsh.model.occ.synchronize()

print("Starting loop over circle ... ")
for i in range(rpam.parameters["N"]):
    Q_x_r = Q(i * delta_theta).dot(x_r)
    Q_x_R = Q(i * delta_theta).dot(x_R)

    p_r = gmsh.model.occ.addPoint(Q_x_r[0], Q_x_r[1], 0)
    p_R = gmsh.model.occ.addPoint(Q_x_R[0], Q_x_R[1], 0)
    gmsh.model.occ.synchronize()

    line_r_R = gmsh.model.occ.addLine(p_r, p_R)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.embed(1, [line_r_R], 2, ring[0][0][0])
    gmsh.model.occ.synchronize()
print("... done.")

# add 2-dimensional objects
surfaces = gmsh.model.occ.getEntities(dim=2)
assert surfaces == ring[0]
disk_subdomain_id = 1

gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], disk_subdomain_id)
gmsh.model.setPhysicalName(surfaces[0][0], disk_subdomain_id, "disk")

# add 1-dimensional objects
lines = gmsh.model.occ.getEntities(dim=1)
circle_r_subdomain_id = 2
circle_R_subdomain_id = 3
# line_p_1_p_2_subdomain_id = 3

gmsh.model.addPhysicalGroup(lines[0][0], [lines[0][1]], circle_r_subdomain_id)
gmsh.model.setPhysicalName(lines[0][0], circle_r_subdomain_id, "circle_r")
gmsh.model.addPhysicalGroup(lines[1][0], [lines[1][1]], circle_R_subdomain_id)
gmsh.model.setPhysicalName(lines[1][0], circle_R_subdomain_id, "circle_R")

# add 0-dimensional objects
vertices = gmsh.model.occ.getEntities(dim=0)

# set the resolution
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [surfaces[0][0]])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "LcMax", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * rpam.parameters["r"])
gmsh.model.mesh.field.setNumber(threshold, "DistMax", rpam.parameters["r"])

circle_r_dist = gmsh.model.mesh.field.add("Distance")
circle_r_threshold = gmsh.model.mesh.field.add("Threshold")

gmsh.model.mesh.field.setNumber(circle_r_threshold, "IField", circle_r_dist)
gmsh.model.mesh.field.setNumber(circle_r_threshold, "LcMin", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(circle_r_threshold, "LcMax", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(circle_r_threshold, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(circle_r_threshold, "DistMax", 0.5)

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, circle_r_threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write(mesh_file)

msh.print_mesh_edges_to_csv(mesh_file, output_directory + 'edges.csv')

mesh_from_file = meshio.read(mesh_file)

msh.full_write(mesh_file, ['triangle', 'line'], metadata, output_directory, True)
