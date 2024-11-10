import numpy
import meshio
import gmsh
import pygmsh
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
args = parser.parse_args()

#mesh resolution
resolution = (float)(args.resolution)

#mesh parameters
#CHANGE PARAMETERS HERE
L = 1.0
h = L
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
          model.add_point((np.pi/8.0, 0, 0), mesh_size=resolution),
          model.add_point((L, 0, 0), mesh_size=resolution)
          ]
my_lines = [model.add_line( points[0], points[1] ), model.add_line( points[1], points[2] )]

#add a 2d object:  a plane surface starting from the 4 lines above


model.synchronize()

print("# of lines added = ", len(my_lines))

model.add_physical([my_lines[0]], "line1")
model.add_physical([my_lines[1]], "line2")
model.add_physical([points[0]], "point_l")
model.add_physical([points[1]], "point_r")

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
# tetra_mesh = create_mesh(mesh_from_file, "tetra", True)
#build a mesh of triangles from the 1d object
# triangle_mesh = create_mesh(mesh_from_file, "triangle", True)
#build a mesh of lines from the 1d object
line_mesh = create_mesh(mesh_from_file, "line", True)


# meshio.write("tetra_mesh.xdmf", tetra_mesh)
# meshio.write("triangle_mesh.xdmf", triangle_mesh)
meshio.write("line_mesh.xdmf", line_mesh)


'''
line_mesh = create_mesh(mesh_from_file, "line", prune_z=False)
meshio.write("line_mesh.xdmf", line_mesh)

triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=False)
meshio.write("triangle_mesh.xdmf", triangle_mesh)
'''