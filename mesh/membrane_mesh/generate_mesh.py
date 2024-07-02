import numpy
import meshio
import gmsh
import pygmsh
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("resolution")
args = parser.parse_args()


#mesh resolution
resolution = (float)(args.resolution)


# Channel parameters
L = 1
h = 1



print("L = ", L)
print("h = ", h)
print("resolution = ", resolution)


# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

my_points = [ model.add_point((0, 0, 0), mesh_size=resolution),
             model.add_point((L, 0, 0), mesh_size=resolution),
             model.add_point((L, h, 0), mesh_size=resolution),
             model.add_point((0, h, 0), mesh_size=resolution)]

channel_lines = [model.add_line(my_points[0], my_points[1]),
                   model.add_line(my_points[1], my_points[2]),
                   model.add_line(my_points[2], my_points[3]),
                   model.add_line(my_points[3], my_points[0])]

channel_loop = model.add_curve_loop(channel_lines)
plane_surface = model.add_plane_surface(channel_loop, holes=[])


model.synchronize()
model.add_physical([plane_surface], "Volume")
model.add_physical([channel_lines[3]], "Inflow")
model.add_physical([channel_lines[1]], "Outflow")
model.add_physical([channel_lines[2]], "TopWall")
model.add_physical([channel_lines[0]], "BottomWall")
# model.add_physical(circle_r.curve_loop.curves, "Circle")

geometry.generate_mesh(dim=2)
gmsh.write("membrane_mesh.msh")
gmsh.clear()
geometry.__exit__()

mesh_from_file = meshio.read("membrane_mesh.msh")

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={
                           "name_to_read": [cell_data]})
    return out_mesh

line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write("line_mesh.xdmf", line_mesh)

triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write("triangle_mesh.xdmf", triangle_mesh)
