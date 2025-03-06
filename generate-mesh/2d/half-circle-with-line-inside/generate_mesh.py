'''
Ths code generates a 2d mesh given by half a circle surface with a line in the surface embedded in the mesh

run with
clear; clear; python3 generate_mesh.py [resolution]
example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py 0.1 $SOLUTION_PATH
'''

import gmsh
import pygmsh
import argparse

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append( module_path )

import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument( "resolution" )
parser.add_argument( "output_directory" )
args = parser.parse_args()

# mesh resolution
resolution = (float)( args.resolution )
mesh_file = args.output_directory + "/mesh.msh"

# mesh parameters
# CHANGE PARAMETERS HERE
r = 1
c_1 = [r, 0, 0]
c_2 = [-r, 0, 0]
c_3 = [r / 2, -r / 8, 0]
c_4 = [-r / 2, -r / 8, 0]
# CHANGE PARAMETERS HERE

print( "r = ", r )
print( "c_1 = ", c_1 )
print( "c_2 = ", c_2 )
print( "resolution = ", resolution )
p_1_id = 1
p_2_id = 2
p_3_id = 6
p_4_id = 7
line_12_id = 3
arc_21_id = 4
surface_id = 5
line_34_id = 8

geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

# add a 0d object:
p_1 = gmsh.model.geo.addPoint( c_1[0], c_1[1], c_1[2] )
p_2 = gmsh.model.geo.addPoint( c_2[0], c_2[1], c_2[2] )
p_c = gmsh.model.geo.addPoint( 0, 0, 0 )
p_3 = gmsh.model.geo.addPoint( c_3[0], c_3[1], c_3[2] )
p_4 = gmsh.model.geo.addPoint( c_4[0], c_4[1], c_4[2] )
gmsh.model.geo.synchronize()

line_12 = gmsh.model.geo.addLine( p_1, p_2 )
gmsh.model.geo.synchronize()

arc_21 = gmsh.model.geo.addCircleArc( p_2, p_c, p_1 )
gmsh.model.geo.synchronize()

line_34 = gmsh.model.geo.addLine( p_3, p_4 )
gmsh.model.geo.synchronize()

loop = gmsh.model.geo.addCurveLoop( [line_12, arc_21] )
gmsh.model.geo.synchronize()

surface = gmsh.model.geo.addPlaneSurface( [loop] )
gmsh.model.geo.synchronize()

gmsh.model.mesh.embed( 1, [line_34], 2, surface )
gmsh.model.geo.synchronize()

# add 0-dimensional objects
vertices = gmsh.model.getEntities( dim=0 )

gmsh.model.addPhysicalGroup( vertices[0][0], [vertices[0][1]], p_1_id )
gmsh.model.setPhysicalName( vertices[0][0], p_1_id, "p_1" )

gmsh.model.addPhysicalGroup( vertices[1][0], [vertices[1][1]], p_2_id )
gmsh.model.setPhysicalName( vertices[1][0], p_2_id, "p_2" )

# add 1-dimensional objects
lines = gmsh.model.getEntities( dim=1 )

gmsh.model.addPhysicalGroup( lines[0][0], [lines[0][1]], line_12_id )
gmsh.model.setPhysicalName( lines[0][0], line_12_id, "line_12" )

gmsh.model.addPhysicalGroup( lines[1][0], [lines[1][1]], arc_21_id )
gmsh.model.setPhysicalName( lines[1][0], arc_21_id, "arc_12" )

gmsh.model.addPhysicalGroup( lines[2][0], [lines[2][1]], line_34_id )
gmsh.model.setPhysicalName( lines[2][0], line_34, "line_34" )

# add 2-dimensional objects
surfaces = gmsh.model.getEntities( dim=2 )

gmsh.model.addPhysicalGroup( surfaces[0][0], [surfaces[0][1]], surface_id )
gmsh.model.setPhysicalName( surfaces[0][0], surface_id, "surface" )

# set the resolution
distance = gmsh.model.mesh.field.add( "Distance" )
gmsh.model.mesh.field.setNumbers( distance, "FacesList", [surface] )

threshold = gmsh.model.mesh.field.add( "Threshold" )
gmsh.model.mesh.field.setNumber( threshold, "IField", distance )
gmsh.model.mesh.field.setNumber( threshold, "LcMin", resolution )
gmsh.model.mesh.field.setNumber( threshold, "LcMax", resolution )
gmsh.model.mesh.field.setNumber( threshold, "DistMin", 0.5 * r )
gmsh.model.mesh.field.setNumber( threshold, "DistMax", r )

minimum = gmsh.model.mesh.field.add( "Min" )
gmsh.model.mesh.field.setNumbers( minimum, "FieldsList", [threshold] )
gmsh.model.mesh.field.setAsBackgroundMesh( minimum )

gmsh.model.geo.synchronize()


geometry.generate_mesh( dim=2 )
gmsh.write( mesh_file )


msh.write_mesh_to_csv( mesh_file, 'solution/line_vertices.csv' )


model.__exit__()



#write mesh components to file
msh.write_mesh_components( mesh_file, "solution/triangle_mesh.xdmf", "triangle", True )
msh.write_mesh_components( mesh_file, "solution/line_mesh.xdmf", "line", True )
msh.write_mesh_components( mesh_file, "solution/vertex_mesh.xdmf", "vertex", True )
