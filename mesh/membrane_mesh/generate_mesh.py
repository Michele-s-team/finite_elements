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
#resolution = 0.3


# Channel parameters
L = 2.2
h = 0.41
r = 0.05
c_r = [0.2, 0.2, 0]
# R = 1.0
# c_R = [0, 0, 0]
#c2 = [0.7, 0.12, 0]
#r = 0.07


print("L = ", L)
print("h = ", h)
# print("r = ", r)
# print("c_r = ", c_r)
print("resolution = ", resolution)


# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()


# Add circle
# circle_r = model.add_circle(c_r, r, mesh_size=resolution/2)
rectangle_Lh = model.add_rectangle(0, L, 0, h, 0, mesh_size=resolution)

# print(circle_r)
print(rectangle_Lh)


# plane_surface = model.add_plane_surface(     rectangle_Lh.curve_loop, holes=[circle_r.curve_loop])
plane_surface = model.add_plane_surface(channel_loop, holes=[])



## Call gmsh kernel before add physical entities
model.synchronize()

model.add_physical([plane_surface], "Volume")
# model.add_physical(circle_r.curve_loop.curves, "Obstacle")

#
## We generate the mesh using the pygmsh function `generate_mesh`. Generate mesh returns a `meshio.Mesh`. However, this mesh is tricky to extract physical tags from. Therefore we write the mesh to file using the `gmsh.write` function.
#
geometry.generate_mesh(64)
gmsh.write("membrane_mesh.msh")
gmsh.clear()
geometry.__exit__()
#
## ## <a name="second"></a>2. How to convert your mesh to XDMF
## Now that we have save the mesh to a `msh` file, we would like to convert it to a format that interfaces with DOLFIN and DOLFINx.
## For this I suggest using the `XDMF`-format as it supports parallel IO.
#
mesh_from_file = meshio.read("membrane_mesh.msh")
#
## Now that we have loaded the mesh, we need to extract the cells and physical data. We need to create a separate file for the facets (lines), which we will use when we define boundary conditions in DOLFIN/DOLFINx. We do this with the following convenience function. Note that as we would like a 2 dimensional mesh, we need to remove the z-values in the mesh coordinates.
#
#
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={
                           "name_to_read": [cell_data]})
    return out_mesh
#
#
## With this function at hand, we can save the meshes to `XDMF`.
#
## +
line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write("line_mesh.xdmf", line_mesh)
#
triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write("triangle_mesh.xdmf", triangle_mesh)
