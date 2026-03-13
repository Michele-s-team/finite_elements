'''
generate a mesh given by a square with a polygon-shaped hole in it

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/polygon"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/polygon/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import gmsh
import meshio
import numpy as np
import os
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import calculus as cal
import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')



# add '/' to output_directory if it is missing
output_directory = io.add_trailing_slash(rarg.args.output_directory)
mesh_file = output_directory + "mesh.msh"

# write into metadata the file format wich which the mesh will be written
mesh_metadata = rpam.parameters.copy()
mesh_metadata['file_format'] = 'xdmf'


print(f'output_directory = "{output_directory}"')

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'



geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()


# add square
square_points = [gmsh.model.geo.addPoint(0, 0, 0),
                gmsh.model.geo.addPoint(rpam.parameters["L"], 0, 0),
                gmsh.model.geo.addPoint(rpam.parameters["L"], rpam.parameters["h"], 0),
                gmsh.model.geo.addPoint(0, rpam.parameters["h"], 0)]

square_lines = [gmsh.model.geo.addLine(square_points[i], square_points[i + 1])
                 for i in range(-1, len(square_points) - 1)]

square_loop = gmsh.model.geo.addCurveLoop(square_lines)


# add polygon
polygon_coordinates = [[0.1, 0.1], [0.7, 0.3], [0.8, 0.4], [0.5, 0.5], [0.3, 0.4]]
polygon_points = [gmsh.model.geo.addPoint(polygon_coordinates[0][0], polygon_coordinates[0][1], 0)]
gmsh.model.geo.synchronize()

polygon_lines = []

for i in range(1, len(polygon_coordinates)):

    polygon_points.append(gmsh.model.geo.addPoint(polygon_coordinates[i][0], polygon_coordinates[i][1], 0))
    gmsh.model.geo.synchronize()

    polygon_lines.append(gmsh.model.geo.addLine(polygon_points[i-1], polygon_points[i]))
    gmsh.model.geo.synchronize()

polygon_lines.append(gmsh.model.geo.addLine(polygon_points[-1], polygon_points[0]))
gmsh.model.geo.synchronize()

polygon_loop = gmsh.model.geo.addCurveLoop(polygon_lines)
gmsh.model.geo.synchronize()

plane_surface = gmsh.model.geo.addPlaneSurface([square_loop, polygon_loop])
gmsh.model.geo.synchronize()


# tag physical objects

# tag 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

# square lines
msh.tag_physical_object(lines[0], rpam.parameters['line_b_id'], gmsh.model, 'line_b')
msh.tag_physical_object(lines[1], rpam.parameters['line_r_id'], gmsh.model, 'line_r')
msh.tag_physical_object(lines[2], rpam.parameters['line_t_id'], gmsh.model, 'line_t')
msh.tag_physical_object(lines[3], rpam.parameters['line_l_id'], gmsh.model, 'line_l')

# polygon lines
msh.tag_physical_object([lines[i] for i in range(4, len(lines))], rpam.parameters['polygon_id'], gmsh.model, 'polygon_line')


# tag 2-dimensional objects
surfaces = gmsh.model.getEntities(dim=2)

msh.tag_physical_object(surfaces[0], rpam.parameters['surface_id'], gmsh.model, 'surface')





# set the mesh resolution
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [polygon_loop])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "LcMax", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", max(rpam.parameters["L"], rpam.parameters["h"]))

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
gmsh.model.geo.synchronize()


geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

msh.full_write(mesh_file, ['triangle', 'line'], mesh_metadata, output_directory, True)

