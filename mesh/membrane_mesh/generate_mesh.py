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
# L = 1.0
# h = 1.0
r = 1.0
c_r = [0, 0, 0]


# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# Add circle
circle_r = model.add_circle(c_r, r, mesh_size=resolution)
# rectangle_Lh = model.add_rectangle(0, L, 0, h, 0, mesh_size=resolution)


# my_points = [ model.add_point((1.3, 1.0, 0), mesh_size=resolution),
#              model.add_point((1.7, 0.7, 0), mesh_size=resolution),
#              model.add_point((2.5, 0.6, 0), mesh_size=resolution),
#              model.add_point((4.2, 1.1, 0), mesh_size=resolution),
#              model.add_point((3.0, 1.3, 0), mesh_size=resolution),
#              model.add_point((1.7, 1.3, 0), mesh_size=resolution),
#              ]


#add the points that define the boundary of the mesh
# points = [model.add_point((0, 0, 0), mesh_size=resolution),
#           model.add_point((L, 0, 0), mesh_size=resolution),
#           model.add_point((L, h, 0), mesh_size=resolution),
#           model.add_point((0, h, 0), mesh_size=resolution)
#           ]


# channel_lines = [model.add_line(points[0], points[1]),
#                    model.add_line(points[1], points[2]),
#                    model.add_line(points[2], points[3]),
#                    model.add_line(points[3], points[0])]


# channel_loop = model.add_curve_loop(channel_lines)

plane_surface = model.add_plane_surface(circle_r.curve_loop, holes=[circle_r.curve_loop])
# plane_surface = model.add_plane_surface(channel_loop, holes=[])



model.synchronize()
model.add_physical([plane_surface], "Volume")

#I will read this tagged element with `ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)`
model.add_physical(circle_r.curve_loop.curves, "Circle")

# model.add_physical([channel_lines[3]], "Inflow")
# model.add_physical([channel_lines[1]], "Outflow")
# model.add_physical([channel_lines[2]], "Top Wall")
# model.add_physical([channel_lines[0]], "Bottom Wall")


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



