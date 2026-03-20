'''
This code generates a 2d mesh (A) given by a square with a polygonal shape (the polygon is meshed inside), plus a one-dimensional mesh (B) given by a line. The line mesh B corresponds to the polygon boundary of A, stretched on a line. 

Here 
    - A is mesh_0
    - B is mesh_1
    
A has 2 sub_meshes 
    - sub_mesh_0_0
    - sub_mesh_0_1
    
and B has no sub_meshes

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/shape/"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/shape/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import colorama as col
from fenics import *
import gmsh
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



# mesh A will be stored in output_directory_square_mesh
output_directory_mesh_0 = io.add_trailing_slash(os.path.join(rarg.args.output_directory, 'mesh_0'))
os.mkdir(output_directory_mesh_0)
# mesh B will be stored in output_directory_line_mesh
output_directory_mesh_1 = io.add_trailing_slash(os.path.join(rarg.args.output_directory, 'mesh_1'))
os.mkdir(output_directory_mesh_1)

mesh_0_file = os.path.join(output_directory_mesh_0, "mesh.msh")

# number of vertices of the polygon boundary
N = len(rpam.parameters['polygon_coordinates'])
# total length of the polygon boundary
polygon_length = cal.polygon_length(rpam.parameters['polygon_coordinates'])

#write metadata for ensemble mesh
mesh_metadata = rpam.parameters.copy()

# write metadata for mesh 0
mesh_0_metadata = {}
mesh_0_metadata['L'] = rpam.parameters['L']
mesh_0_metadata['h'] = rpam.parameters['h']
mesh_0_metadata['resolution'] = rpam.parameters['resolution']
mesh_0_metadata['n_sub_meshes'] = rpam.parameters['n_sub_meshes_0']
mesh_0_metadata['polygon_coordinates'] = rpam.parameters['polygon_coordinates']

mesh_0_metadata['sub_mesh_0_dim'] = rpam.parameters['sub_mesh_0_0_dim']
mesh_0_metadata['sub_mesh_1_dim'] = rpam.parameters['sub_mesh_0_1_dim']

mesh_0_metadata['sub_mesh_0_id'] = rpam.parameters['sub_mesh_0_0_id']
mesh_0_metadata['sub_mesh_1_id'] = rpam.parameters['sub_mesh_0_1_id']

mesh_0_metadata['line_l_id'] = rpam.parameters['line_l_id']
mesh_0_metadata['line_r_id'] = rpam.parameters['line_r_id']
mesh_0_metadata['line_t_id'] = rpam.parameters['line_t_id']
mesh_0_metadata['line_b_id'] = rpam.parameters['line_b_id']
mesh_0_metadata['polygon_id'] = rpam.parameters['polygon_id']

mesh_0_metadata['file_format'] = 'xdmf'


# write metadata for mesh 1
mesh_1_metadata = {}

mesh_1_metadata['L'] = polygon_length
mesh_1_metadata['x_l'] = 0
mesh_1_metadata['x_r'] = mesh_1_metadata['L']
mesh_1_metadata['N'] = N
mesh_1_metadata['polygon_coordinates'] = rpam.parameters['polygon_coordinates']


mesh_1_metadata['vertex_l_id'] = rpam.parameters['vertex_l_id']
mesh_1_metadata['vertex_r_id'] = rpam.parameters['vertex_r_id']
mesh_1_metadata['line_id'] = rpam.parameters['polygon_id']

mesh_1_metadata['file_format'] = 'h5'

geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()


# A) generate mesh A (square with circle)

#1. add  square


square_p_bl = gmsh.model.geo.addPoint(0, 0, 0)
square_p_br = gmsh.model.geo.addPoint(rpam.parameters["L"], 0, 0)
square_p_tr = gmsh.model.geo.addPoint(rpam.parameters["L"], rpam.parameters["h"], 0)
square_p_tl = gmsh.model.geo.addPoint(0, rpam.parameters["h"], 0)
gmsh.model.geo.synchronize()

square_line_b = gmsh.model.geo.addLine(square_p_bl, square_p_br)
square_line_r = gmsh.model.geo.addLine(square_p_br, square_p_tr)
square_line_t = gmsh.model.geo.addLine(square_p_tr, square_p_tl)
square_line_l = gmsh.model.geo.addLine(square_p_tl, square_p_bl)
gmsh.model.geo.synchronize()

square_loop = gmsh.model.geo.addCurveLoop([square_line_b, square_line_r, square_line_t, square_line_l])
gmsh.model.geo.synchronize()


#2. add polygon


polygon_points = [gmsh.model.geo.addPoint(rpam.parameters['polygon_coordinates'][0][0], rpam.parameters['polygon_coordinates'][0][1], 0)]
gmsh.model.geo.synchronize()

polygon_lines = []

print(f'Added point with coordinates {rpam.parameters["polygon_coordinates"][-1]}')

print("Starting loop over polygon ... ")

for i in range(1, N):

    polygon_points.append(gmsh.model.geo.addPoint(rpam.parameters['polygon_coordinates'][i][0], rpam.parameters['polygon_coordinates'][i][1], 0))
    gmsh.model.geo.synchronize()

    polygon_lines.append(gmsh.model.geo.addLine(polygon_points[-2], polygon_points[-1]))
    gmsh.model.geo.synchronize()

print("... done.")

polygon_lines.append(gmsh.model.geo.addLine(polygon_points[-1], polygon_points[0]))
gmsh.model.geo.synchronize()



polygon_loop = gmsh.model.geo.addCurveLoop(polygon_lines)
gmsh.model.geo.synchronize()

square_minus_polygon_surface = gmsh.model.geo.addPlaneSurface([square_loop, polygon_loop])
gmsh.model.geo.synchronize()

gmsh.model.mesh.embed(1, polygon_lines, 2, square_minus_polygon_surface)
gmsh.model.geo.synchronize()

polygon_surface = gmsh.model.geo.addPlaneSurface([polygon_loop])
gmsh.model.geo.synchronize()



# add 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

# add square lines
msh.tag_physical_object(lines[0], rpam.parameters['line_b_id'], gmsh.model, 'line_b')
msh.tag_physical_object(lines[1], rpam.parameters['line_r_id'], gmsh.model, 'line_r')
msh.tag_physical_object(lines[2], rpam.parameters['line_t_id'], gmsh.model, 'line_t')
msh.tag_physical_object(lines[3], rpam.parameters['line_l_id'], gmsh.model, 'line_l')

#add polygon lines
msh.tag_physical_object([lines[i] for i in range(4, 4 + N)], rpam.parameters['polygon_id'], gmsh.model, 'polygon_loop')



# add 2-dimensional objects
surfaces = gmsh.model.getEntities(dim=2)

msh.tag_physical_object(surfaces[0], rpam.parameters['sub_mesh_0_1_id'], gmsh.model, 'square_minus_polygon_surface')
msh.tag_physical_object(surfaces[1], rpam.parameters['sub_mesh_0_0_id'], gmsh.model, 'polygon_surface')


# set the resolution
# se resolution equal to parameters["resolution"] at a distance 0 from surface_in, and  at distance max(rpam.parameters["L"],rpam.parameters["h"]) from sub_mesh_0_1_id
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
gmsh.write(mesh_0_file)

msh.full_write(mesh_0_file, ['triangle', 'line'], mesh_0_metadata, output_directory_mesh_0, True)

msh.generate_sub_mesh(output_directory_mesh_0, os.path.join(output_directory_mesh_0, 'sub_meshes', 'sub_mesh_0'), rpam.parameters["sub_mesh_0_0_id"])
msh.generate_sub_mesh(output_directory_mesh_0, os.path.join(output_directory_mesh_0, 'sub_meshes', 'sub_mesh_1'), rpam.parameters["sub_mesh_0_1_id"])


# print the boundary points of the boundary given by the polygon
msh.sorted_boundary_points(
    msh.read_mesh(os.path.join(output_directory_mesh_0, 'triangle_mesh.xdmf')), 
    output_directory_mesh_0, 
    [rpam.parameters['polygon_id']],
    os.path.join(output_directory_mesh_0, 'boundary_points_id_' + str(rpam.parameters['polygon_id']) + '.csv'))


# check that the number of mesh vertices on the circle matches N and if it does not, abort. 
mesh_0 = msh.read_mesh(os.path.join(output_directory_mesh_0, 'triangle_mesh.xdmf'))
mf_mesh_0 = msh.read_mesh_components(mesh_0, mesh_0.topology().dim() - 1, os.path.join(output_directory_mesh_0, 'line_mesh.xdmf'))

# collect unique vertex indices touched by facets tagged with polygon_id
polygon_vertex_ids = set()

for facet in facets(mesh_0):
    #run through all facets of mesh_0 

    if mf_mesh_0[facet] == rpam.parameters['polygon_id']:
        # the facet under consideration belongs to the circle

        for v in vertices(facet):
            # run through the vertices of the facet under consideration, and ad them to polygon_vertex_ids

            polygon_vertex_ids.add(v.index())

n_vertices_on_polygon = len(polygon_vertex_ids)
print(f'Number of vertices on polygon = {n_vertices_on_polygon}')


# B) mesh B (line)


# generate the line mesh corresponding to the polygon
msh.genereate_line_mesh(0, polygon_length, N,
                        rpam.parameters['polygon_id'], rpam.parameters['vertex_l_id'], rpam.parameters['vertex_r_id'],
                        x_m=None,
                        vertex_m_id=None,
                        output_directory=output_directory_mesh_1, 
                        metadata=mesh_1_metadata)



#print overall mesh metadata
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'mesh_metadata.csv'), mesh_metadata)

model.__exit__()