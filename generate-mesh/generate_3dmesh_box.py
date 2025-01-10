'''
This code generates a 3d mesh given by a box

run with
clear; clear; python3 generate_3dmesh_box.py [resolution]
example:
clear; clear; rm -r solution; mkdir solution; python3 generate_3dmesh_box.py 0.1
'''

import meshio
import gmsh
import pygmsh
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
args = parser.parse_args()

#mesh resolution
resolution = (float)(args.resolution)
print("resolution = ", resolution)

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.occ.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

#generate a box
#
#CHANGE PARAMETERS HERE
L1 = 1
L2 = 0.5
L3 = 0.45
#CHANGE PARAMETERS HERE
box = model.add_box([0, 0, 0], [L1, L2, L3], mesh_size=resolution)
model.synchronize()
model.add_physical([box], "box")
#


geometry.generate_mesh(dim=3)
gmsh.write("solution/mesh.msh")
model.__exit__()

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={
                           cell_type: cells}, cell_data={"name_to_read": [cell_data]})
    return out_mesh


mesh_from_file = meshio.read("solution/mesh.msh")

#create a tetrahedron mesh
tetrahedron_mesh = create_mesh(mesh_from_file, "tetra", True)
meshio.write("solution/tetrahedron_mesh.xdmf", tetrahedron_mesh)
