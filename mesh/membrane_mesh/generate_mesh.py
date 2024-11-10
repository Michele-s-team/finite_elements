import numpy
import meshio
import gmsh
import pygmsh
import argparse
# from dolfin import *

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
args = parser.parse_args()

#mesh resolution
resolution = (float)(args.resolution)

#mesh parameters
#CHANGE PARAMETERS HERE
L = 1.0
h = 1.0
r = 1.0
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

#add a 1d object a set of lines
points = [model.add_point( (0, 0, 0), mesh_size=resolution ),
          model.add_point((L, 0, 0), mesh_size=resolution),
          model.add_point((L, h, 0), mesh_size=resolution),
          model.add_point((0, h, 0), mesh_size=resolution)]
channel_lines = [model.add_line(points[i], points[i+1])
                  for i in range(-1, len(points)-1)]
channel_loop = model.add_curve_loop(channel_lines)

#add a 2d object:  a plane surface starting from the 4 lines above
plane_surface = model.add_plane_surface(channel_loop, holes=[])

#add a 3d object: a ball
ball = model.add_ball(c_r, r)

model.synchronize()

model.add_physical([ball], "ball")
model.add_physical([plane_surface], "surface")
model.add_physical([channel_lines[0]], "line")

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

#build a mesh of tetrahedra from the 3d object
tetra_mesh = create_mesh(mesh_from_file, "tetra", True)
#build a mesh of triangles from the 1d object
triangle_mesh = create_mesh(mesh_from_file, "triangle", True)
#build a mesh of lines from the 1d object
line_mesh = create_mesh(mesh_from_file, "line", True)


meshio.write("tetra_mesh.xdmf", tetra_mesh)
meshio.write("triangle_mesh.xdmf", triangle_mesh)
meshio.write("line_mesh.xdmf", line_mesh)


'''
line_mesh = create_mesh(mesh_from_file, "line", prune_z=False)
meshio.write("line_mesh.xdmf", line_mesh)

triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=False)
meshio.write("triangle_mesh.xdmf", triangle_mesh)
'''