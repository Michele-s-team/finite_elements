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

import calc as cal
import mesh as msh
import geometry as geo

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

theta = 2 * np.pi / N
phi = theta/2
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

# close the loop with a special curve addition
arc_12 = gmsh.model.occ.addCircleArc( p_1, p_c_r, p_1_start )
line_23 = gmsh.model.occ.addLine( p_1_start, p_4_start )
arc_34 = gmsh.model.occ.addCircleArc( p_4_start, p_c_R, p_4 )
line_41 = gmsh.model.occ.addLine( p_4, p_1 )
gmsh.model.occ.synchronize()

loop = gmsh.model.occ.addCurveLoop( [arc_12, line_23, arc_34, line_41] )
surfaces.append( gmsh.model.occ.addPlaneSurface( [loop] ) )
gmsh.model.occ.synchronize()

# add 2-dimensional objects
# # surfaces = gmsh.model.occ.getEntities(dim=2)
# assert surfaces == surface
surface_tot_subdomain_id = 1

gmsh.model.addPhysicalGroup( 2, surfaces, surface_tot_subdomain_id )
gmsh.model.setPhysicalName( 2, surface_tot_subdomain_id, "square" )

# add 1-dimensional objects
lines = gmsh.model.occ.getEntities( dim=1 )
# arc_12_id = 2
# arc_34_id = 3

# loop through all surfaces and find the surface in the i-th slice according to its center of mass (COM):
# the COM of slices in ds_r will have |COM|<r, the COM of slices in ds_R will have r < |COM| < R. The COM of radial lines will have |COM| = (r+R)/2
print(f"center of mass of first slice: {Q(  theta + theta / 2 ).dot( np.array( [r, 0] ) )}")

for line in lines:
    # compute the center of mass of each surface, and recognize according to the coordinates of the center of mass
    center_of_mass = gmsh.model.occ.getCenterOfMass( line[0], line[1] )
    r_s = r * np.sin(phi) * (1  - np.sin(phi)**3/3 - np.cos(phi)**2)/(phi - np.sin(phi) * np.cos(phi))

    com_r = [center_of_mass[0], center_of_mass[1]]
    # print( f"|center of mass|: {geo.my_norm(com_r)}" )



    if(geo.my_norm(com_r) < r):
        print(f"line belongs to ds_r, angle = {cal.atan_quad(com_r)}")
        for i in range(N):
            if(np.isclose(cal.atan_quad(com_r) ,(theta/2 + theta * i), 1e-2)):
                print(f"line has i = {i}")


    if((geo.my_norm(com_r) < R) & (geo.my_norm(com_r) > (r+R)/2)):
        print(f"line belongs to ds_R, angle = {cal.atan_quad(com_r)}")

    # for i in range( N ):
    #     if np.allclose( [center_of_mass[0], center_of_mass[1]], Q( i * theta + theta / 2 ).dot( np.array( [r, 0] ) ) ):

# gmsh.model.addPhysicalGroup( lines[0][0], [lines[0][1]], arc_12_id )
# gmsh.model.setPhysicalName( lines[0][0], arc_12_id, "arc_12" )
#
# gmsh.model.addPhysicalGroup( lines[2][0], [lines[2][1]], arc_34_id )
# gmsh.model.setPhysicalName( lines[2][0], arc_34_id, "arc_34" )
#
#
#
#
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

# # create a line mesh
#
# line_mesh = create_mesh(mesh_from_file, "line", True)
# meshio.write("solution/line_mesh.xdmf", line_mesh)
#

'''
#create a vertex mesh
vertex_mesh = create_mesh(mesh_from_file, "vertex", True)
meshio.write("solution/vertex_mesh.xdmf", vertex_mesh)
print(f"Check if all line vertices are triangle vertices : {np.isin( msh.line_vertices( msh_file_path ), msh.triangle_vertices( msh_file_path ) ) }")
'''
