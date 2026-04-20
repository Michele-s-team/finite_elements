'''
This code generates a 2d mesh given by a disk. The disk boundary is given by a polygon with N vertices, where N is determined from the mesh resolution. The polygon, disk surface and vertices are all tagged as separate physical entities. Vertices are tagged with IDs rpam.parameters["vertex_0_id"], rpam.parameters["vertex_0_id"]+1, ....  starting from vertex with coordinate [rpam.parameters['r'], 0] and continuing along the polygon in the counterclockwise direction

The vertex is tagged as a one-dimensional mesh component. 

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/disk/vertices/"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/disk/vertices/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
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


mesh_file = os.path.join(rarg.args.output_directory, "mesh.msh")

'''
The outer circle is created as a polygon 

The number of segments in the polygon is chosen in such a way that the side of the circle polygon is (at the most) equal to the mesh resolution: 

2 * r * sin(2 * pi / N / 2) = resolution
pi / N = arcsin(resolution / (2 r))
N = pi / arcsin(resolution / (2 r))

thus I set
'''

N = int(np.ceil(np.pi / np.arcsin(rpam.parameters['resolution']/ (2.0 * rpam.parameters['r']))))

print(f'N = {N}')

# angular fraction corresponding to each segment of the circle
delta_theta = 2 * np.pi / N


#write metadata for ensemble mesh
mesh_metadata = rpam.parameters.copy()
mesh_metadata['file_format'] = 'xdmf'
mesh_metadata['N'] = N

geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()



# add polygon (circle)
circle_coordinates = [[rpam.parameters['r'], 0]]
circle_vertices = [gmsh.model.geo.addPoint(circle_coordinates[0][0], circle_coordinates[0][1], 0)]
gmsh.model.geo.synchronize()


circle_lines = []

print("Starting loop over circle ... ")
for i in range(1, N):

    circle_coordinates.append(cal.R(i * delta_theta).dot(circle_coordinates[0]))

    circle_vertices.append(gmsh.model.geo.addPoint(circle_coordinates[-1][0], circle_coordinates[-1][1], 0))
    gmsh.model.geo.synchronize()

    circle_lines.append(gmsh.model.geo.addLine(circle_vertices[-2], circle_vertices[-1]))
    gmsh.model.geo.synchronize()

print("... done.")

circle_lines.append(gmsh.model.geo.addLine(circle_vertices[-1], circle_vertices[0]))
gmsh.model.geo.synchronize()

circle_loop = gmsh.model.geo.addCurveLoop(circle_lines)
gmsh.model.geo.synchronize()

circle_surface = gmsh.model.geo.addPlaneSurface([circle_loop])
gmsh.model.geo.synchronize()


gmsh.model.mesh.embed(1, circle_lines, 2, circle_surface)
gmsh.model.geo.synchronize()


# add 0-dimensional objects
vertex_list = gmsh.model.getEntities(dim=0)
#add circle vertex_list: each vertex is added with a different ID
for i in range(N):
    gmsh.model.addPhysicalGroup(vertex_list[i][0], [vertex_list[i][1]], rpam.parameters["vertex_0_id"] + i)
    gmsh.model.setPhysicalName(vertex_list[i][0], rpam.parameters["vertex_0_id"] + i, f"vertex_{i}")


# add 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

#add circle lines
gmsh.model.addPhysicalGroup(1, [lines[i][1] for i in range(0, N)], rpam.parameters["circle_id"])
gmsh.model.setPhysicalName(1, rpam.parameters["circle_id"], "circle_loop")


# add 2-dimensional objects
surfaces = gmsh.model.getEntities(dim=2)

gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], rpam.parameters["surface_id"])
gmsh.model.setPhysicalName(surfaces[0][0], rpam.parameters["surface_id"], "disk_surface")




# set the resolution
# se resolution equal to parameters["resolution"] at a distance 0 from surface_in, and  at distance max(rpam.parameters["L"],rpam.parameters["h"]) from sub_mesh_0_1_id
distance = gmsh.model.mesh.field.add("Distance")

gmsh.model.mesh.field.setNumbers(distance, "FacesList", [circle_loop])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "LcMax", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", rpam.parameters["r"])


minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

gmsh.model.geo.synchronize()

geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

msh.full_write(mesh_file, ['triangle', 'line', 'vertex'], mesh_metadata, rarg.args.output_directory, True)


# check that the number of mesh vertices on the circle matches N and if it does not, abort. 
read_mesh = msh.read_mesh(os.path.join(rarg.args.output_directory, 'triangle_mesh.xdmf'))
mf_mesh_0 = msh.read_mesh_components(read_mesh, read_mesh.topology().dim() - 1, os.path.join(rarg.args.output_directory, 'line_mesh.xdmf'))

# collect unique vertex indices touched by facets tagged with circle_id
circle_vertex_ids = set()

for facet in facets(read_mesh):
    #run through all facets of mesh_0 

    if mf_mesh_0[facet] == rpam.parameters['circle_id']:
        # the facet under consideration belongs to the circle

        for v in vertices(facet):
            # run through the vertices of the facet under consideration, and ad them to circel_vertex_ids

            circle_vertex_ids.add(v.index())

n_vertices_on_circle = len(circle_vertex_ids)

if n_vertices_on_circle != N:
    # the meshing algorithm has added additional vertices on the circle, while I want the number of vertices on the circle to match N, and thus the number of vertices in the line mesh -> print an error message

    print(f"{col.Fore.RED}{'Error: the number of vertices on circle does not match the number of vertices of the 1d mesh!!! Aborting...'}{col.Style.RESET_ALL}")

    sys.exit()


#print overall mesh metadata
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'mesh_metadata.csv'), mesh_metadata)

model.__exit__()
