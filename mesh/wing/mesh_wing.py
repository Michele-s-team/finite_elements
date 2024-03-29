
# +
import numpy as np
import meshio
import gmsh
import pygmsh
resolution = 0.08
# Channel parameters
L = 10.
H = 2.
#scaling factor of the wing
s = 0.5
#rotation angle of the wing with respect to the z axis
theta = np.radians(0)
#the center about which the wing will be rotated
c = [L/4.0, H/2.0]

cos = np.cos(theta)
sin = np.sin(theta)
R = [[cos,-sin],[sin,cos]]
print("Rotation matrix: ",R)


# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

#points defining the coordinates of the non-rotated wing
r=[[1.3, 1.0], [1.7, 0.7], [2.5, 0.6], [4.2, 1.1], [3.0, 1.3], [1.7, 1.3]]
for i in range(0, len(r)):
    r[i] = c + np.multiply(np.dot(R, np.subtract(r[i], c)), s)

# wing profile
my_points = [model.add_point([r[i][0], r[i][1], 0.0], mesh_size=resolution)
                 for i in range(0, len(r))]

# my_points = [ model.add_point(r[0], mesh_size=resolution),
#              model.add_point(r[1], mesh_size=resolution),
#              model.add_point(r[2], mesh_size=resolution),
#              model.add_point(r[3], mesh_size=resolution),
#              model.add_point(r[4], mesh_size=resolution),
#              model.add_point(r[5], mesh_size=resolution),
#              ]

# print(np.dot(R, r))

# print(circle_R)
# print(circle_r)

# print("Points defining the wing:")
for point in my_points:
    print(point)


my_spline = model.add_spline([my_points[0], my_points[1], my_points[2], my_points[3], my_points[4], my_points[5], my_points[0]])
#spline2 = model.add_spline([my_points[2], my_points[3], my_points[0]])

my_loop = model.add_curve_loop([my_spline])

# -

# The next step is to create the channel with the circle as a hole.

# +
# Add points with finer resolution on left side
points = [model.add_point((0, 0, 0), mesh_size=resolution),
          model.add_point((L, 0, 0), mesh_size=5*resolution),
          model.add_point((L, H, 0), mesh_size=5*resolution),
          model.add_point((0, H, 0), mesh_size=resolution)]


# #
## Add lines between all points creating the rectangle
channel_lines = [model.add_line(points[i], points[i+1])
                 for i in range(-1, len(points)-1)]

# channel_lines = [arc_R_in_up, arc_R_in_down, model.add_line(p3, p4), arc_R_out_down, arc_R_out_up, model.add_line(p6,p1)]


print("channel lines : ")
print(channel_lines)
# #
## Create a line loop and plane surface for meshing
channel_loop = model.add_curve_loop(channel_lines)
#spline_loop = model.add_curve_loop(spline)
# my_surface = model.add_plane_surface(my_loop)

# plane_surface = model.add_plane_surface(     circle_R.curve_loop, holes=[circle_r.curve_loop])
# plane_surface = model.add_plane_surface(channel_loop, holes=[my_loop])
plane_surface = model.add_plane_surface(channel_loop, holes=[my_loop])


## Call gmsh kernel before add physical entities
model.synchronize()
## -
#
## The final step before mesh generation is to mark the different boundaries and the volume mesh. Note that with pygmsh, boundaries with the same tag has to be added simultaneously. In this example this means that we have to add the top and
##  bottom wall in one function call.
#
volume_marker = 4
model.add_physical([plane_surface], "Volume")
model.add_physical(channel_lines[0], "Inflow")
model.add_physical(channel_lines[2], "Outflow")
model.add_physical([channel_lines[1], channel_lines[3]], "Walls")
# model.add_physical(circle_r.curve_loop.curves, "Obstacle")
#model.add_physical(circle2.curve_loop.curves, "Obstacle 2")

#
## We generate the mesh using the pygmsh function `generate_mesh`. Generate mesh returns a `meshio.Mesh`. However, this mesh is tricky to extract physical tags from. Therefore we write the mesh to file using the `gmsh.write` function.
#
geometry.generate_mesh(dim=2)
gmsh.write("mesh_wing.msh")
gmsh.clear()
geometry.__exit__()
#
mesh_from_file = meshio.read("mesh_wing.msh")
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
