'''
This code generates a 3d mesh given by a ring
To see the values of subdomain_id assigned to each tagged element, see read_2dmesh_ring.py and the comments in this file

run with
clear; clear; python3 generate_2dmesh_ring.py [resolution]
example:
clear; clear; rm -r solution; mkdir solution; python3 generate_2dmesh_ring.py 0.1
'''

import meshio
import argparse
import numpy as np
import gmsh
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
args = parser.parse_args()

warnings.filterwarnings("ignore")
gmsh.initialize()

gmsh.model.add("my model")
c_r = [0, 0, 0]
c_R = [0, 0, 0]
r = 1
R=2
resolution = (float)(args.resolution)
print(f"Mesh resolution = {resolution}")




disk_r = gmsh.model.occ.addDisk( c_r[0], c_r[1], c_r[2], r, r )
disk_R = gmsh.model.occ.addDisk( c_R[0], c_R[1], c_R[2], R, R )
ring = gmsh.model.occ.cut( [(2, disk_R)], [(2, disk_r)] )

p_1 = gmsh.model.occ.addPoint((r+R)/2.0, 0, 0)
p_2 = gmsh.model.occ.addPoint((r+R)/2.0, r/10.0, 0)
line_p_1_p_2 = gmsh.model.occ.addLine(p_1, p_2)

gmsh.model.occ.synchronize()



surfaces = gmsh.model.occ.getEntities(dim=2)
assert surfaces == ring[0]
disk_subdomain_id = 0
gmsh.model.addPhysicalGroup( surfaces[0][0], [surfaces[0][1]], disk_subdomain_id )
gmsh.model.setPhysicalName( surfaces[0][0], disk_subdomain_id, "disk" )

lines = gmsh.model.occ.getEntities(dim=1)

print(f"lines = {lines}")
print(f"line_p1_p_2 = {line_p_1_p_2}")
#these are the subdomain_ids with which the components will be read in read_2dmesh_ring.py
circle_r_subdomain_id = 1
circle_R_subdomain_id = 2
line_p_1_p_2_subdomain_id = 3

gmsh.model.addPhysicalGroup( lines[0][0], [lines[0][1]], circle_r_subdomain_id )
gmsh.model.setPhysicalName( lines[0][0], disk_subdomain_id, "circle_r" )

gmsh.model.addPhysicalGroup( lines[1][0], [lines[1][1]], circle_R_subdomain_id )
gmsh.model.setPhysicalName( lines[1][0], disk_subdomain_id, "circle_R" )

gmsh.model.addPhysicalGroup( lines[2][0], [lines[2][1]], line_p_1_p_2_subdomain_id )
gmsh.model.setPhysicalName( lines[2][0], line_p_1_p_2_subdomain_id, "line_p_1_p_2" )




#set the resolution close to circle_r
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [surfaces[0][0]])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", resolution)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * r)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", r)

circle_r_dist = gmsh.model.mesh.field.add("Distance")
circle_r_threshold = gmsh.model.mesh.field.add( "Threshold" )

gmsh.model.mesh.field.setNumber( circle_r_threshold, "IField", circle_r_dist )
gmsh.model.mesh.field.setNumber( circle_r_threshold, "LcMin", resolution )
gmsh.model.mesh.field.setNumber( circle_r_threshold, "LcMax", resolution )
gmsh.model.mesh.field.setNumber( circle_r_threshold, "DistMin", 0.1 )
gmsh.model.mesh.field.setNumber( circle_r_threshold, "DistMax", 0.5 )

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers( minimum, "FieldsList", [threshold, circle_r_threshold] )
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)


gmsh.model.occ.synchronize()


gmsh.model.mesh.generate(2)

gmsh.write("solution/mesh.msh")


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={
                           cell_type: cells}, cell_data={"name_to_read": [cell_data]})
    return out_mesh


mesh_from_file = meshio.read("solution/mesh.msh")


#create a triangle mesh in which the surfaces will be stored
triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=False)
meshio.write("solution/triangle_mesh.xdmf", triangle_mesh)

#create a line mesh
line_mesh = create_mesh(mesh_from_file, "line", True)
meshio.write("solution/line_mesh.xdmf", line_mesh)
