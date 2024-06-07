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
print("r = ", r)
print("c_r = ", c_r)
print("resolution = ", resolution)


# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()


# Add circle
# circle_R = model.add_circle(c_R, R, mesh_size=resolution)
circle_r = model.add_circle(c_r, r, mesh_size=resolution/2)
rectangle_Lh = model.add_rectangle(0, L, 0, h, 0, mesh_size=resolution)

# o_in = geometry.add_point([-L/2,0,0])
# o_out = geometry.add_point([L/2,0,0])


# p1 = geometry.add_point([-L/2,R,0])
# p2 = geometry.add_point([-L/2-R,0,0])
# p3 = geometry.add_point([-L/2,-R,0])

# arc_R_in_up = model.add_circle_arc(p1,o_in,p2)
# arc_R_in_down = model.add_circle_arc(p2,o_in,p3)

# p4 = geometry.add_point([L/2,-R,0])
# p5 = geometry.add_point([L/2+R,0,0])
# p6 = geometry.add_point([L/2,R,0])

# arc_R_out_down = model.add_circle_arc(p4,o_out,p5)
# arc_R_out_up = model.add_circle_arc(p5,o_out,p6)


#
# my_points = [ model.add_point((1.3, 1.0, 0), mesh_size=resolution),
#              model.add_point((1.7, 0.7, 0), mesh_size=resolution),
#              model.add_point((2.5, 0.6, 0), mesh_size=resolution),
#              model.add_point((4.2, 1.1, 0), mesh_size=resolution),
#              model.add_point((3.0, 1.3, 0), mesh_size=resolution),
#              model.add_point((1.7, 1.3, 0), mesh_size=resolution),
#              ]

# print(circle_R)
# print(circle_r)
print(rectangle_Lh)

# print("Points defining the spline:")
# for point in my_points:
#     print(point)
             
# spline1 = model.add_spline([my_points[0], my_points[1], my_points[2], my_points[3], my_points[4], my_points[5], my_points[0]])
#spline2 = model.add_spline([my_points[2], my_points[3], my_points[0]])

# my_loop = model.add_curve_loop([spline1])

# -

# The next step is to create the channel with the circle as a hole.

# +
# Add points with finer resolution on left side
# points = [model.add_point((-L/2, -h/2, 0), mesh_size=resolution),
#           model.add_point((L/2, -h/2, 0), mesh_size=5*resolution),
#           model.add_point((L/2, h/2, 0), mesh_size=5*resolution),
#           model.add_point((-L/2, h/2, 0), mesh_size=resolution)]

# Add lines between all points creating the rectangle
# channel_lines = [model.add_line(points[i], points[i+1])
#                  for i in range(-1, len(points)-1)]

# channel_lines = [arc_R_in_up, arc_R_in_down, model.add_line(p3, p4), arc_R_out_down, arc_R_out_up, model.add_line(p6,p1)]


# print("channel lines : ")
# print(channel_lines)
# #
## Create a line loop and plane surface for meshing
# channel_loop = model.add_curve_loop(channel_lines)
#spline_loop = model.add_curve_loop(spline)
# my_surface = model.add_plane_surface(my_loop)

plane_surface = model.add_plane_surface(     rectangle_Lh.curve_loop, holes=[circle_r.curve_loop])
# plane_surface = model.add_plane_surface(channel_loop, holes=[circle_r.curve_loop])



## Call gmsh kernel before add physical entities
model.synchronize()
## -
#
## The final step before mesh generation is to mark the different boundaries and the volume mesh. Note that with pygmsh, boundaries with the same tag has to be added simultaneously. In this example this means that we have to add the top and
##  bottom wall in one function call.
#
#volume_marker = 6
model.add_physical([plane_surface], "Volume")
# model.add_physical(arc_R_in_up, "Inflow")
# model.add_physical(arc_R_out, "Outflow")
# model.add_physical([channel_lines[1], channel_lines[3]], "Walls")
model.add_physical(circle_r.curve_loop.curves, "Obstacle")
#model.add_physical(circle2.curve_loop.curves, "Obstacle 2")

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
