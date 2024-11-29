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

# parameters
r = 1.0
R = 2.0
c_r = [0, 0, 0]
c_R = [0, 0, 0]

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# Add circle
circle_r = model.add_circle(c_r, r, mesh_size=resolution)
circle_R = model.add_circle(c_R, R, mesh_size=resolution)

plane_surface = model.add_plane_surface(circle_R.curve_loop, holes=[circle_r.curve_loop])

model.synchronize()
model.add_physical([plane_surface], "Volume")

#I will read this tagged element with `ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)`
model.add_physical(circle_r.curve_loop.curves, "Circle r")
model.add_physical(circle_R.curve_loop.curves, "Circle R")

geometry.generate_mesh(64)
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