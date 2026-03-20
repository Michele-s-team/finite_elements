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
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
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

'''
# write metadata for mesh 1
mesh_1_metadata = {}

mesh_1_metadata['L'] = N * rpam.parameters['r'] * 2.0 * np.sin(delta_theta/2.0)
mesh_1_metadata['x_l'] = 0
mesh_1_metadata['x_r'] = mesh_1_metadata['L']
mesh_1_metadata['N'] = N

mesh_1_metadata['vertex_l_id'] = rpam.parameters['vertex_l_id']
mesh_1_metadata['vertex_r_id'] = rpam.parameters['vertex_r_id']
mesh_1_metadata['line_id'] = rpam.parameters['polygon_id']

mesh_1_metadata['file_format'] = 'h5'


print("output_directory = ", rarg.args.output_directory)

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


#2. add circle


circle_coordinates = [np.array([rpam.parameters["c_r"][0] + rpam.parameters['r'], rpam.parameters['c_r'][1]])]
circle_points = [gmsh.model.geo.addPoint(circle_coordinates[0][0], circle_coordinates[0][1], 0)]
gmsh.model.geo.synchronize()

circle_lines = []

print(f'Added point with coordinates {circle_coordinates[-1]}')


print("Starting loop over circle ... ")
for i in range(1, N):

    circle_coordinates.append(
        np.add(rpam.parameters['c_r'], cal.R(i * delta_theta).dot(np.subtract(circle_coordinates[0], rpam.parameters['c_r'])))
        )

    circle_points.append(gmsh.model.geo.addPoint(circle_coordinates[-1][0], circle_coordinates[-1][1], 0))
    gmsh.model.geo.synchronize()

    circle_lines.append(gmsh.model.geo.addLine(circle_points[-2], circle_points[-1]))
    gmsh.model.geo.synchronize()

print("... done.")

circle_lines.append(gmsh.model.geo.addLine(circle_points[-1], circle_points[0]))
gmsh.model.geo.synchronize()



circle_loop = gmsh.model.geo.addCurveLoop(circle_lines)
gmsh.model.geo.synchronize()

square_minus_circle_surface = gmsh.model.geo.addPlaneSurface([square_loop, circle_loop])
gmsh.model.geo.synchronize()

gmsh.model.mesh.embed(1, circle_lines, 2, square_minus_circle_surface)
gmsh.model.geo.synchronize()

circle_surface = gmsh.model.geo.addPlaneSurface([circle_loop])
gmsh.model.geo.synchronize()



# add 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

# add square lines
gmsh.model.addPhysicalGroup(lines[0][0], [lines[0][1]], rpam.parameters["line_b_id"])
gmsh.model.setPhysicalName(lines[0][0], rpam.parameters["line_b_id"], "square_line_b")

gmsh.model.addPhysicalGroup(lines[1][0], [lines[1][1]], rpam.parameters["line_r_id"])
gmsh.model.setPhysicalName(lines[1][0], rpam.parameters["line_r_id"], "square_line_r")

gmsh.model.addPhysicalGroup(lines[2][0], [lines[2][1]], rpam.parameters["line_t_id"])
gmsh.model.setPhysicalName(lines[2][0], rpam.parameters["line_t_id"], "square_line_t")

gmsh.model.addPhysicalGroup(lines[3][0], [lines[3][1]], rpam.parameters["line_l_id"])
gmsh.model.setPhysicalName(lines[3][0], rpam.parameters["line_l_id"], "square_line_l")

#add circle lines
gmsh.model.addPhysicalGroup(1, [lines[i][1] for i in range(4, 4 + N)], rpam.parameters["polygon_id"])
gmsh.model.setPhysicalName(1, rpam.parameters["polygon_id"], "circle_loop")


# add 2-dimensional objects
surfaces = gmsh.model.getEntities(dim=2)

gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], rpam.parameters["sub_mesh_0_1_id"])
gmsh.model.setPhysicalName(surfaces[0][0], rpam.parameters["sub_mesh_0_1_id"], "square_minus_circle_surface")

gmsh.model.addPhysicalGroup(surfaces[1][0], [surfaces[1][1]], rpam.parameters["sub_mesh_0_0_id"])
gmsh.model.setPhysicalName(surfaces[1][0], rpam.parameters["sub_mesh_0_0_id"], "circle_surface")



# set the resolution
# se resolution equal to parameters["resolution"] at a distance 0 from surface_in, and  at distance max(rpam.parameters["L"],rpam.parameters["h"]) from sub_mesh_0_1_id
distance = gmsh.model.mesh.field.add("Distance")

gmsh.model.mesh.field.setNumbers(distance, "FacesList", [circle_loop])

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


# print the boundary points of the boundary given by the circle
msh.sorted_boundary_points(
    msh.read_mesh(os.path.join(output_directory_mesh_0, 'triangle_mesh.xdmf')), 
    output_directory_mesh_0, 
    [rpam.parameters['polygon_id']],
    os.path.join(output_directory_mesh_0, 'boundary_points_id_' + str(rpam.parameters['polygon_id']) + '.csv'))


# check that the number of mesh vertices on the circle matches N and if it does not, abort. 
mesh_0 = msh.read_mesh(os.path.join(output_directory_mesh_0, 'triangle_mesh.xdmf'))
mf_mesh_0 = msh.read_mesh_components(mesh_0, mesh_0.topology().dim() - 1, os.path.join(output_directory_mesh_0, 'line_mesh.xdmf'))

# collect unique vertex indices touched by facets tagged with polygon_id
circle_vertex_ids = set()

for facet in facets(mesh_0):
    #run through all facets of mesh_0 

    if mf_mesh_0[facet] == rpam.parameters['polygon_id']:
        # the facet under consideration belongs to the circle

        for v in vertices(facet):
            # run through the vertices of the facet under consideration, and ad them to circel_vertex_ids

            circle_vertex_ids.add(v.index())

n_vertices_on_circle = len(circle_vertex_ids)
print(f'Number of vertices on circle = {n_vertices_on_circle}')

if n_vertices_on_circle != N:
    # the meshing algorithm has added additional vertices on the circle, while I want the number of vertices on the circle to match N, and thus the number of vertices in the line mesh -> print an error message

    print(f"{col.Fore.RED}{'Error: the number of vertices on circle does not match the number of vertices of the 1d mesh!!! Aborting...'}{col.Style.RESET_ALL}")

    sys.exit()






# B) mesh B (line)


# generate the line mesh corresponding to the circle
msh.genereate_line_mesh(0, N * rpam.parameters['r'] * 2.0 * np.sin(delta_theta/2.0), N,
                        rpam.parameters['polygon_id'], rpam.parameters['vertex_l_id'], rpam.parameters['vertex_r_id'],
                        x_m=None,
                        vertex_m_id=None,
                        output_directory=output_directory_mesh_1, 
                        metadata=mesh_1_metadata)



#print overall mesh metadata
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'mesh_metadata.csv'), mesh_metadata)

model.__exit__()

'''