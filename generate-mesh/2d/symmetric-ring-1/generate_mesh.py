'''
This code generates a 2d mesh (given by a ring between circles with radii r and R ) with perfect spherical symmetry by dividing the ring into N slices and replicating each slide.
dx is tagged with id = 2*N
each ds in circle_r is tagged with id = 0, ..., N-1
each ds in circle_r is tagged with id = N, ..., 2*N-1


run with
clear; clear; python3 generate_mesh.py.py [resolution] [output directory]
example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py 0.1 $SOLUTION_PATH

'''

import meshio
import numpy as np
import gmsh
import argparse
from fenics import *

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append( module_path )

import geometry as geo
import input_output as io
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument( "resolution" )
parser.add_argument( "output_directory" )
args = parser.parse_args()

msh_file_path = (args.output_directory) + "/mesh.msh"
gmsh.initialize()
gmsh.model.add( "my model" )

c_r = [0, 0, 0]
c_R = [0, 0, 0]
r = 1
R = 2
N = 128
resolution = (float)( args.resolution )
print( f"Mesh resolution = {resolution}" )

theta = 2 * np.pi / N
phi = theta / 2
epsilon = 1e-2
p_c_r = gmsh.model.occ.addPoint( c_r[0], c_r[1], 0 )
p_c_R = gmsh.model.occ.addPoint( c_R[0], c_R[1], 0 )


def Q(theta):
    return np.array( [[np.cos( theta ), -np.sin( theta )], [np.sin( theta ), np.cos( theta )]] )


# initialize the loop over 0 <= theta < 2 pi
r_1 = np.array( [r, 0] )
r_2 = Q( theta ).dot( r_1 )
r_4 = np.array( [R, 0] )
r_3 = Q( theta ).dot( r_4 )

p_1 = gmsh.model.occ.addPoint( r_1[0], r_1[1], 0 )
p_2 = gmsh.model.occ.addPoint( r_2[0], r_2[1], 0 )
p_3 = gmsh.model.occ.addPoint( r_3[0], r_3[1], 0 )
p_4 = gmsh.model.occ.addPoint( r_4[0], r_4[1], 0 )
p_1_start = p_1
p_4_start = p_4
gmsh.model.occ.synchronize()

surfaces = []

# loop through N-1 slices of the ring
for i in range( N - 1 ):
    print( f"Adding slice #{i} ... " )

    print( f"\tr_1 = {r_1}" )
    print( f"\tr_2 = {r_2}" )
    print( f"\tr_3 = {r_3}" )
    print( f"\tr_4 = {r_4}" )

    arc_12 = gmsh.model.occ.addCircleArc( p_1, p_c_r, p_2 )
    line_23 = gmsh.model.occ.addLine( p_2, p_3 )
    arc_34 = gmsh.model.occ.addCircleArc( p_3, p_c_R, p_4 )
    line_41 = gmsh.model.occ.addLine( p_4, p_1 )
    gmsh.model.occ.synchronize()

    loop = gmsh.model.occ.addCurveLoop( [arc_12, line_23, arc_34, line_41] )
    surfaces.append( gmsh.model.occ.addPlaneSurface( [loop] ) )
    gmsh.model.occ.synchronize()

    r_2 = Q( theta ).dot( r_2 )
    r_3 = Q( theta ).dot( r_3 )

    p_1 = p_2
    p_2 = gmsh.model.occ.addPoint( r_2[0], r_2[1], 0 )
    p_4 = p_3
    p_3 = gmsh.model.occ.addPoint( r_3[0], r_3[1], 0 )
    gmsh.model.occ.synchronize()

    print( "...done" )

# close the loop with a special curve addition for the last slice
arc_12 = gmsh.model.occ.addCircleArc( p_1, p_c_r, p_1_start )
line_23 = gmsh.model.occ.addLine( p_1_start, p_4_start )
arc_34 = gmsh.model.occ.addCircleArc( p_4_start, p_c_R, p_4 )
line_41 = gmsh.model.occ.addLine( p_4, p_1 )
gmsh.model.occ.synchronize()

loop = gmsh.model.occ.addCurveLoop( [arc_12, line_23, arc_34, line_41] )
surfaces.append( gmsh.model.occ.addPlaneSurface( [loop] ) )
gmsh.model.occ.synchronize()

# add 2-dimensional objects
# surfaces = gmsh.model.occ.getEntities(dim=2)
# assert surfaces == surface
surface_tot_subdomain_id = 2 * N

gmsh.model.addPhysicalGroup( 2, surfaces, surface_tot_subdomain_id )
gmsh.model.setPhysicalName( 2, surface_tot_subdomain_id, "ring" )

# add 1-dimensional objects
lines = gmsh.model.occ.getEntities( dim=1 )

# loop through all lines and find whether the line belongs to the circle with radius r or to the circle with radius R by looking at the center of mass (COM) of the line
id_r = 0
id_R = N
for line in lines:
    # compute the center of mass of each line, and recognize the line according to the coordinates of the center of mass
    center_of_mass = gmsh.model.occ.getCenterOfMass( line[0], line[1] )

    x_com = [center_of_mass[0], center_of_mass[1]]

    if (geo.my_norm( x_com ) < r):
        gmsh.model.addPhysicalGroup( line[0], [line[1]], id_r )
        gmsh.model.setPhysicalName( line[0], id_r, "arc_" + str( id_r ) )
        id_r += 1

    if ((geo.my_norm( x_com ) < R) & (geo.my_norm( x_com ) > (r + R) / 2 + epsilon)):
        gmsh.model.addPhysicalGroup( line[0], [line[1]], id_R )
        gmsh.model.setPhysicalName( line[0], id_R, "arc_" + str( id_R ) )
        id_R += 1

# #add 0-dimensional objects
# '''
# vertices = gmsh.model.occ.getEntities(dim=0)
# p_1_subdomain_id = 4
# p_2_subdomain_id = 5
# p_3_subdomain_id = 6
# p_4_subdomain_id = 7
# pp_2_subdomain_id = 8
# pp_3_subdomain_id = 9
#
# gmsh.model.addPhysicalGroup( vertices[2][0], [vertices[2][1]], p_1_subdomain_id )
# gmsh.model.setPhysicalName( vertices[2][0], p_1_subdomain_id, "p_1" )
#
# gmsh.model.addPhysicalGroup( vertices[3][0], [vertices[3][1 ]], p_2_subdomain_id )
# gmsh.model.setPhysicalName( vertices[3][0], p_2_subdomain_id, "p_2" )
#
# gmsh.model.addPhysicalGroup( vertices[4][0], [vertices[4][1 ]], p_3_subdomain_id )
# gmsh.model.setPhysicalName( vertices[4][0], p_3_subdomain_id, "p_3" )
#
# gmsh.model.addPhysicalGroup( vertices[5][0], [vertices[5][1 ]], p_4_subdomain_id )
# gmsh.model.setPhysicalName( vertices[5][0], p_4_subdomain_id, "p_4" )
#
# gmsh.model.addPhysicalGroup( vertices[6][0], [vertices[6][1 ]], pp_2_subdomain_id )
# gmsh.model.setPhysicalName( vertices[6][0], pp_2_subdomain_id, "pp_2" )
#
# gmsh.model.addPhysicalGroup( vertices[7][0], [vertices[7][1 ]], pp_3_subdomain_id )
# gmsh.model.setPhysicalName( vertices[7][0], pp_3_subdomain_id, "pp_3" )
#
# '''

# set the resolution
distance = gmsh.model.mesh.field.add( "Distance" )
gmsh.model.mesh.field.setNumbers( distance, "FacesList", surfaces )

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

# create triangle
triangle_mesh = msh.create_mesh( mesh_from_file, "triangle", prune_z=True )
meshio.write( (args.output_directory) + "/triangle_mesh.xdmf", triangle_mesh )

# create line mesh
line_mesh = msh.create_mesh( mesh_from_file, "line", True )
meshio.write( (args.output_directory) + "/line_mesh.xdmf", line_mesh )

'''
#create  vertex mesh
vertex_mesh = create_mesh(mesh_from_file, "vertex", True)
meshio.write((args.output_directory) + "/vertex_mesh.xdmf", vertex_mesh)
print(f"Check if all line vertices are triangle vertices : {np.isin( msh.line_vertices( msh_file_path ), msh.triangle_vertices( msh_file_path ) ) }")
'''

# print the mesh vertices to file
mesh = msh.read_mesh( args.output_directory + "/triangle_mesh.xdmf" )
io.print_vertices_to_csv_file( mesh, (args.output_directory) + "/vertices.csv" )
