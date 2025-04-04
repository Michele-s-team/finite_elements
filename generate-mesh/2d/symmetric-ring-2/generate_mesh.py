'''
This code generates a 3d mesh given by a ring
To see the values of subdomain_id assigned to each tagged element, see read_mesh.py and the comments in this file

run with
clear; clear; python3 generate_mesh.py [resolution]
example:
clear; clear; rm -r solution; mkdir solution; python3 generate_mesh.py 0.1
'''

import meshio
import argparse
import numpy as np
import gmsh
import warnings
from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append( module_path )

import input_output as io
import mesh as msh


parser = argparse.ArgumentParser()
parser.add_argument( "resolution" )
args = parser.parse_args()

msh_file_path = "solution/mesh.msh"

warnings.filterwarnings( "ignore" )
gmsh.initialize()

gmsh.model.add( "my model" )
c_r = [0, 0, 0]
c_R = [0, 0, 0]
r = 1
R = 2
N = 64
resolution = (float)( args.resolution )
print( f"Mesh resolution = {resolution}" )

delta_theta = 2 * np.pi / N

def Q(theta):
    return np.array( [[np.cos( theta ), -np.sin( theta )], [np.sin( theta ), np.cos( theta )]] )

x_r = np.array( [r+resolution, 0] )
x_R = np.array( [R-resolution, 0] )


disk_r = gmsh.model.occ.addDisk( c_r[0], c_r[1], c_r[2], r, r )
disk_R = gmsh.model.occ.addDisk( c_R[0], c_R[1], c_R[2], R, R )
# add this every time you add a component to the mesh and every time you make modifications to the mesh
gmsh.model.occ.synchronize()

ring = gmsh.model.occ.cut( [(2, disk_R)], [(2, disk_r)] )
gmsh.model.occ.synchronize()

print("Starting loop over circle ... ")
for i in range(N):

    Q_x_r = Q( i * delta_theta ).dot( x_r )
    Q_x_R = Q( i * delta_theta ).dot( x_R )

    p_r = gmsh.model.occ.addPoint( Q_x_r[0], Q_x_r[1], 0 )
    p_R = gmsh.model.occ.addPoint( Q_x_R[0], Q_x_R[1], 0 )
    gmsh.model.occ.synchronize()

    line_r_R = gmsh.model.occ.addLine( p_r, p_R )
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.embed( 1, [line_r_R], 2, ring[0][0][0] )
    gmsh.model.occ.synchronize()
print("... done.")

# add 2-dimensional objects
surfaces = gmsh.model.occ.getEntities( dim=2 )
assert surfaces == ring[0]
disk_subdomain_id = 1

gmsh.model.addPhysicalGroup( surfaces[0][0], [surfaces[0][1]], disk_subdomain_id )
gmsh.model.setPhysicalName( surfaces[0][0], disk_subdomain_id, "disk" )

# add 1-dimensional objects
lines = gmsh.model.occ.getEntities( dim=1 )
circle_r_subdomain_id = 2
circle_R_subdomain_id = 3
# line_p_1_p_2_subdomain_id = 3

gmsh.model.addPhysicalGroup( lines[0][0], [lines[0][1]], circle_r_subdomain_id )
gmsh.model.setPhysicalName( lines[0][0], circle_r_subdomain_id, "circle_r" )
gmsh.model.addPhysicalGroup( lines[1][0], [lines[1][1]], circle_R_subdomain_id )
gmsh.model.setPhysicalName( lines[1][0], circle_R_subdomain_id, "circle_R" )

# add 0-dimensional objects
vertices = gmsh.model.occ.getEntities( dim=0 )

# set the resolution
distance = gmsh.model.mesh.field.add( "Distance" )
gmsh.model.mesh.field.setNumbers( distance, "FacesList", [surfaces[0][0]] )

threshold = gmsh.model.mesh.field.add( "Threshold" )
gmsh.model.mesh.field.setNumber( threshold, "IField", distance )
gmsh.model.mesh.field.setNumber( threshold, "LcMin", resolution )
gmsh.model.mesh.field.setNumber( threshold, "LcMax", resolution )
gmsh.model.mesh.field.setNumber( threshold, "DistMin", 0.5 * r )
gmsh.model.mesh.field.setNumber( threshold, "DistMax", r )

circle_r_dist = gmsh.model.mesh.field.add( "Distance" )
circle_r_threshold = gmsh.model.mesh.field.add( "Threshold" )

gmsh.model.mesh.field.setNumber( circle_r_threshold, "IField", circle_r_dist )
gmsh.model.mesh.field.setNumber( circle_r_threshold, "LcMin", resolution )
gmsh.model.mesh.field.setNumber( circle_r_threshold, "LcMax", resolution )
gmsh.model.mesh.field.setNumber( circle_r_threshold, "DistMin", 0.1 )
gmsh.model.mesh.field.setNumber( circle_r_threshold, "DistMax", 0.5 )

minimum = gmsh.model.mesh.field.add( "Min" )
gmsh.model.mesh.field.setNumbers( minimum, "FieldsList", [threshold, circle_r_threshold] )
gmsh.model.mesh.field.setAsBackgroundMesh( minimum )

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate( 2 )
gmsh.write( msh_file_path )





mesh_from_file = meshio.read( msh_file_path )

# create a triangle mesh in which the surfaces will be stored
triangle_mesh = msh.create_mesh( mesh_from_file, "triangle", prune_z=True )
meshio.write( "solution/triangle_mesh.xdmf", triangle_mesh )

# create a line mesh
line_mesh = msh.create_mesh( mesh_from_file, "line", True )
meshio.write( "solution/line_mesh.xdmf", line_mesh )

# create a vertex mesh
'''
vertex_mesh = msh.create_mesh( mesh_from_file, "vertex", True )
meshio.write( "solution/vertex_mesh.xdmf", vertex_mesh )
print( f"Check if all line vertices are triangle vertices : {np.isin( msh.line_vertices( msh_file_path ), msh.triangle_vertices( msh_file_path ) )}" )
'''

#print the mesh vertices to file
mesh = msh.read_mesh("solution/triangle_mesh.xdmf")
io.print_vertices_to_csv_file(mesh, "solution/vertices.csv" )