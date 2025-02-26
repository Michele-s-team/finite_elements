'''
This code generates a 3d mesh given by a ring
To see the values of subdomain_id assigned to each tagged element, see read_2dmesh_ring.py and the comments in this file

run with
clear; clear; python3 generate_2dmesh_ring_fraction.py [resolution]
example:
clear; clear; rm -r solution; mkdir solution; python3 generate_2dmesh_ring_fraction.py 0.1
'''

import meshio
import argparse
import numpy as np
import gmsh
import warnings

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append( module_path )

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
N = 8
resolution = (float)( args.resolution )
print( f"Mesh resolution = {resolution}" )

theta = 2.0 * np.pi / N
p_c_r = gmsh.model.occ.addPoint( c_r[0], c_r[1], 0 )
p_c_R = gmsh.model.occ.addPoint( c_R[0], c_R[1], 0 )


def Q(theta):
    return np.array( [[np.cos( theta ), -np.sin( theta )], [np.sin( theta ), np.cos( theta )]] )


######## add first slice ########
r_1 = np.array( [0, r] )
r_2 = Q( -theta ).dot( r_1 )
r_4 = np.array( [0, R] )
r_3 = Q( -theta ).dot( r_4 )

print( f"r_1 = {r_1}" )
print( f"r_2 = {r_2}" )
print( f"r_3 = {r_3}" )
print( f"r_4 = {r_4}" )

p_1 = gmsh.model.occ.addPoint( r_1[0], r_1[1], 0 )
p_2 = gmsh.model.occ.addPoint( r_2[0], r_2[1], 0 )
p_3 = gmsh.model.occ.addPoint( r_3[0], r_3[1], 0 )
p_4 = gmsh.model.occ.addPoint( r_4[0], r_4[1], 0 )
gmsh.model.occ.synchronize()

arc_12 = gmsh.model.occ.addCircleArc( p_1, p_c_r, p_2 )
line_34 = gmsh.model.occ.addLine( p_2, p_3 )
arc_34 = gmsh.model.occ.addCircleArc( p_3, p_c_R, p_4 )
line_61 = gmsh.model.occ.addLine( p_4, p_1 )
gmsh.model.occ.synchronize()

loop = gmsh.model.occ.addCurveLoop( [arc_12, line_34, arc_34, line_61] )
surface = gmsh.model.occ.addPlaneSurface( [loop] )
gmsh.model.occ.synchronize()

# add 2-dimensional objects
# surfaces = gmsh.model.occ.getEntities(dim=2)
# assert surfaces == surface
surface_subdomain_id = 6

gmsh.model.addPhysicalGroup( 2, [surface], surface_subdomain_id )
gmsh.model.setPhysicalName( 2, surface_subdomain_id, "square" )


######## add second slice ########
r_2 = Q( -theta ).dot( r_2 )
r_3 = Q( -theta ).dot( r_3 )

p_1 = p_2
p_2 = gmsh.model.occ.addPoint( r_2[0], r_2[1], 0 )
p_4 = p_3
p_3 = gmsh.model.occ.addPoint( r_3[0], r_3[1], 0 )
gmsh.model.occ.synchronize()

arc_12 = gmsh.model.occ.addCircleArc( p_1, p_c_r, p_2 )
line_34 = gmsh.model.occ.addLine( p_2, p_3 )
arc_34 = gmsh.model.occ.addCircleArc( p_3, p_c_R, p_4 )
line_61 = gmsh.model.occ.addLine( p_4, p_1 )
gmsh.model.occ.synchronize()



'''
#add 1-dimensional objects
lines = gmsh.model.occ.getEntities(dim=1)
circle_r_subdomain_id = 1
circle_R_subdomain_id = 2
line_p_1_p_2_subdomain_id = 3

gmsh.model.addPhysicalGroup( lines[0][0], [lines[0][1]], circle_r_subdomain_id )
gmsh.model.setPhysicalName( lines[0][0], circle_r_subdomain_id, "circle_r" )

gmsh.model.addPhysicalGroup( lines[1][0], [lines[1][1]], circle_R_subdomain_id )
gmsh.model.setPhysicalName( lines[1][0], circle_R_subdomain_id, "circle_R" )

gmsh.model.addPhysicalGroup( 1, [line_12], line_p_1_p_2_subdomain_id )
gmsh.model.setPhysicalName(1, line_p_1_p_2_subdomain_id, "line_p_1_p_2")


#add 0-dimensional objects
vertices = gmsh.model.occ.getEntities(dim=0)
p_1_subdomain_id = 4
p_2_subdomain_id = 5

gmsh.model.addPhysicalGroup( vertices[0][0], [vertices[0][1]], p_1_subdomain_id )
gmsh.model.setPhysicalName( vertices[0][0], p_1_subdomain_id, "p_1" )

gmsh.model.addPhysicalGroup( vertices[1][0], [vertices[1][1]], p_2_subdomain_id )
gmsh.model.setPhysicalName( vertices[1][0], p_2_subdomain_id, "p_2" )


'''
# set the resolution
distance = gmsh.model.mesh.field.add( "Distance" )
gmsh.model.mesh.field.setNumbers( distance, "FacesList", [surface] )

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


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type( cell_type )
    cell_data = mesh.get_cell_data( "gmsh:physical", cell_type )
    out_mesh = meshio.Mesh( points=mesh.points, cells={
        cell_type: cells}, cell_data={"name_to_read": [cell_data]} )
    return out_mesh


mesh_from_file = meshio.read( msh_file_path )

# create a triangle mesh in which the surfaces will be stored
triangle_mesh = create_mesh( mesh_from_file, "triangle", prune_z=True )
meshio.write( "solution/triangle_mesh.xdmf", triangle_mesh )

# create a line mesh
''' 
line_mesh = create_mesh(mesh_from_file, "line", True)
meshio.write("solution/line_mesh.xdmf", line_mesh)
'''

'''
#create a vertex mesh
vertex_mesh = create_mesh(mesh_from_file, "vertex", True)
meshio.write("solution/vertex_mesh.xdmf", vertex_mesh)
print(f"Check if all line vertices are triangle vertices : {np.isin( msh.line_vertices( msh_file_path ), msh.triangle_vertices( msh_file_path ) ) }")
'''
