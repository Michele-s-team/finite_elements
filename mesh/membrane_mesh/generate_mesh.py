import numpy
import meshio
import gmsh
import pygmsh
import argparse
from dolfin import *


parser = argparse.ArgumentParser()
parser.add_argument("resolution")
args = parser.parse_args()


#mesh resolution
resolution = (float)(args.resolution)


# Channel parameters
#CHANGE PARAMETERS HERE
L = 1.0
h = 1.0
r = 0.0
c_r = [0, 0, 0]
#CHANGE PARAMETERS HERE


print("L = ", L)
print("h = ", h)
print("r = ", r)
print("c_r = ", c_r)
print("resolution = ", resolution)


# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.occ.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

ball = model.add_ball([0, 0, 0], 1)

model.synchronize()

model.add_physical([ball], "ball")

geometry.generate_mesh(dim=3)
gmsh.write("mesh.msh")
model.__exit__()

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={
                           cell_type: cells}, cell_data={"name_to_read": [cell_data]})
    return out_mesh


mesh_from_file = meshio.read("mesh.msh")

tetra_mesh = create_mesh(mesh_from_file, "tetra", True)
# triangle_mesh = create_mesh(mesh_from_file, "triangle", True)
# line_mesh = create_mesh(mesh_from_file, "line", True)


'''
line_mesh = create_mesh(mesh_from_file, "line", prune_z=False)
meshio.write("line_mesh.xdmf", line_mesh)

triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=False)
meshio.write("triangle_mesh.xdmf", triangle_mesh)
'''