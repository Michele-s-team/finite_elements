'''
generate a mesh given by a square whose top line is a one-dimensional submesh

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
    
Here 'sub_mesh_0' is the two-dimensional square mesh and 'sub_mesh_1' is the one-dimensional top edge of the square.
'''

from fenics import *
import gmsh
import math
import os
import pygmsh
import sys
import numpy as np
# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')

output_directory = io.add_trailing_slash(rarg.args.output_directory)
sub_mesh_1_output_directory = io.add_trailing_slash(output_directory + 'sub_mesh_1')
mesh_file = output_directory + "mesh.msh"
mesh_metadata_file_name = output_directory + 'mesh_metadata.csv'

metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'
# print("METADATA KEYS BEFORE WRITE:")
# print(metadata.keys())
# print("n_sub_meshes =", metadata['n_sub_meshes'])
# print("output_directory = ", output_directory)

geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

# -----------------------------------------------------------------------
# Cosine parameters
# -----------------------------------------------------------------------
A    = rpam.parameters["A"]     # amplitude
lmda = rpam.parameters["lmda"] # wavelength
L    = rpam.parameters["L"]
h    = rpam.parameters["h"]
n    = rpam.parameters["n"]


# cosine shape function
def cosine_y(x):
    return h + A * np.cos(n*np.pi* x/ lmda )


nPoints = 3000  # number of discretisation point

# left endpoint  (x=0)
p_top_start = gmsh.model.geo.addPoint(0, cosine_y(0), 0)
pointList_top = [p_top_start]
for i in range(1, nPoints):
    x = L * i / nPoints
    pointList_top.append(gmsh.model.geo.addPoint(x, cosine_y(x), 0))
# right endpoint (x=L)
p_top_end = gmsh.model.geo.addPoint(L, cosine_y(L), 0)
pointList_top.append(p_top_end)

spline_top = gmsh.model.geo.addSpline(pointList_top)   # sub_mesh_1


bottom_points = []
for i in range(1, nPoints):
    x = i*L/(nPoints-1)
    bottom_points.append(
        gmsh.model.geo.addPoint(
            x, A*np.cos(n*np.pi*x/lmda),0
        )
    )

# -----------------------------------------------------------------------
# 2 straight sides
# -----------------------------------------------------------------------
p_bl = bottom_points[0]  # bottom-left
p_br = bottom_points[-1]   # bottom-right

line_12 = gmsh.model.geo.addSpline(bottom_points)              # bottom  (sub_mesh_2)
line_23 = gmsh.model.geo.addLine(p_br, p_top_end)          # right wall
line_41 = gmsh.model.geo.addLine(p_top_start, p_bl)        # left  wall

# CCW loop: bottom → right → cosine (reversed) → left
loop = gmsh.model.geo.addCurveLoop([line_12, line_23, -spline_top, line_41])
surface_domain = gmsh.model.geo.addPlaneSurface([loop])
gmsh.model.geo.synchronize()

# -----------------------------------------------------------------------
# Physical groups – using geo kernel
# -----------------------------------------------------------------------
print(f"DEBUG: spline_top={spline_top}, line_12={line_12}, line_23={line_23}, line_41={line_41}")

gmsh.model.geo.addPhysicalGroup(1, [line_12],    rpam.parameters["sub_mesh_2_id"])
gmsh.model.setPhysicalName    (1, rpam.parameters["sub_mesh_2_id"],        "sub_mesh_2")

gmsh.model.geo.addPhysicalGroup(1, [line_23],    rpam.parameters["line_sub_mesh_0_r_id"])
gmsh.model.setPhysicalName    (1, rpam.parameters["line_sub_mesh_0_r_id"], "line_23")

gmsh.model.geo.addPhysicalGroup(1, [spline_top], rpam.parameters["sub_mesh_1_id"])
gmsh.model.setPhysicalName    (1, rpam.parameters["sub_mesh_1_id"],        "sub_mesh_1")

gmsh.model.geo.addPhysicalGroup(1, [line_41],    rpam.parameters["line_sub_mesh_0_l_id"])
gmsh.model.setPhysicalName    (1, rpam.parameters["line_sub_mesh_0_l_id"], "line_41")
print("line_12 =", line_12)
print("line_23 =", line_23)
print("line_41 =", line_41)
print("spline_top =", spline_top)
print("sub_mesh_0_id =", rpam.parameters["sub_mesh_0_id"])
print("sub_mesh_1_id =", rpam.parameters["sub_mesh_1_id"])
print("sub_mesh_2_id =", rpam.parameters["sub_mesh_2_id"])

gmsh.model.geo.addPhysicalGroup(2, [surface_domain], rpam.parameters["sub_mesh_0_id"])
gmsh.model.setPhysicalName    (2, rpam.parameters["sub_mesh_0_id"], "sub_mesh_0")

print(f"DEBUG: Physical group IDs:")
print(f"  sub_mesh_0_id (surface):      {rpam.parameters['sub_mesh_0_id']}")
print(f"  sub_mesh_1_id (cosine spline): {rpam.parameters['sub_mesh_1_id']}")


gmsh.model.geo.synchronize()

dist_field = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", [spline_top])
gmsh.model.mesh.field.setNumber (dist_field, "NumPointsPerCurve", 1000)

thr_field = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(thr_field, "InField",  dist_field)
gmsh.model.mesh.field.setNumber(thr_field, "SizeMin",  rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(thr_field, "SizeMax",  rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(thr_field, "DistMin",  0.0)
gmsh.model.mesh.field.setNumber(thr_field, "DistMax",  max(L, h))

min_field = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [thr_field])
gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)

gmsh.model.mesh.generate(2)
print("Physical groups:")
for dim, tag in gmsh.model.getPhysicalGroups():
    print(dim, tag,
          gmsh.model.getEntitiesForPhysicalGroup(dim, tag))
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

gmsh.write(mesh_file)
metadata['n_sub_meshes'] = 3
metadata['sub_mesh_2_dim'] = 1
metadata['sub_mesh_2_id'] = rpam.parameters['sub_mesh_2_id']
msh.full_write(mesh_file, ['triangle', 'line'], metadata, output_directory, True)

# print the boundary points of the boundaries given by the top line (sub_mesh 1)
msh.sorted_boundary_points(
    msh.read_mesh(os.path.join(output_directory, 'triangle_mesh.xdmf')), 
    output_directory, 
    [rpam.parameters['sub_mesh_1_id']],
    os.path.join(output_directory, 'boundary_points_id_' + str(rpam.parameters['sub_mesh_1_id']) + '.csv'))



model.__exit__()

# ========================================================================
# Generate submesh for the top edge from the 2D mesh and save it in .h5 format
# ========================================================================

print("Generating H5 sub_mesh for top edge from 2D mesh...")

# Read the generated 2D mesh from the triangle component file
mesh_temp = Mesh()
with XDMFFile(output_directory + "triangle_mesh.xdmf") as infile:
    infile.read(mesh_temp)


# collect vertices that lie on the cosine top boundary
top_edge_vertices = []
for vertex in vertices(mesh_temp):
    point = vertex.point()
    x, y = point.x(), point.y()
    y_cos = h + A * math.cos(n * math.pi * x / lmda)
    if math.isclose(y, y_cos, abs_tol=rpam.parameters["resolution"] * 1e-3):
        top_edge_vertices.append(x)

''' create a list of the vertices in mesh_2d which lies on the bottom edge-analogus to the top membrane '''
bottom_edge_vertices = []

for vertex in vertices(mesh_temp):
    point = vertex.point()

    y_bottom = A*np.cos(n*np.pi*point.x()/lmda)

    if math.isclose(point.y(), y_bottom, rel_tol=1e-4, abs_tol=1e-4):
        bottom_edge_vertices.append(point.x())

bottom_edge_vertices = sorted(list(set(bottom_edge_vertices)))
print("BOTTOM EDGE VERTICES:")
print(bottom_edge_vertices)
print("NUMBER OF BOTTOM VERTICES =", len(bottom_edge_vertices))


# Sort vertices by x-coordinate and remove duplicates
top_edge_vertices = sorted(list(set(top_edge_vertices)))

print(f"Found {len(top_edge_vertices)} unique vertices on top edge")

# Create a proper 1D IntervalMesh using the actual vertex positions
if len(top_edge_vertices) >= 2:

    num_intervals = len(top_edge_vertices) - 1

    # Create output directory for submesh
    sub_mesh_1_output_directory = output_directory + "sub_meshes/1/"
    os.makedirs(sub_mesh_1_output_directory, exist_ok=True)

    sub_mesh_1_metadata = dict([])
    sub_mesh_1_metadata['x_l'] = 0.0
    sub_mesh_1_metadata['x_r'] = rpam.parameters['L']
    sub_mesh_1_metadata['coordinates'] = top_edge_vertices
    sub_mesh_1_metadata['A'] = A
    sub_mesh_1_metadata['lmda'] = lmda
    sub_mesh_1_metadata['resolution'] = rpam.parameters['resolution']
    sub_mesh_1_metadata['line_id'] = rpam.parameters['sub_mesh_1_id']
    sub_mesh_1_metadata['vertex_l_id'] = rpam.parameters['vertex_sub_mesh_1_l_id']
    sub_mesh_1_metadata['vertex_r_id'] = rpam.parameters['vertex_sub_mesh_1_r_id']
    sub_mesh_1_metadata['file_format'] = 'h5'

    # generate the line mesh with the specific coordinates written in top_edge_vertices, which may not be equally spaced
    msh.genereate_line_mesh(0.0, rpam.parameters['L'], num_intervals,
                            rpam.parameters['sub_mesh_1_id'], rpam.parameters['vertex_sub_mesh_1_l_id'], rpam.parameters['vertex_sub_mesh_1_r_id'],
                            output_directory=sub_mesh_1_output_directory, metadata=sub_mesh_1_metadata,
                            coordinates=top_edge_vertices)

    print("===== GENERATING SUBMESH 2 =====")
    print("Number of bottom vertices =", len(bottom_edge_vertices))
    print(bottom_edge_vertices[:10])

    
    sub_mesh_2_output_directory = output_directory + "sub_meshes/2/"
    os.makedirs(sub_mesh_2_output_directory, exist_ok=True)

    sub_mesh_2_metadata = dict([])
    sub_mesh_2_metadata['x_l'] = 0.0
    sub_mesh_2_metadata['x_r'] = rpam.parameters['L']
    sub_mesh_2_metadata['coordinates'] = bottom_edge_vertices
    sub_mesh_2_metadata['resolution'] = rpam.parameters['resolution']
    sub_mesh_2_metadata['line_id'] = rpam.parameters['sub_mesh_2_id']
    sub_mesh_2_metadata['vertex_l_id'] = rpam.parameters['vertex_sub_mesh_2_l_id']
    sub_mesh_2_metadata['vertex_r_id'] = rpam.parameters['vertex_sub_mesh_2_r_id']
    sub_mesh_2_metadata['file_format'] = 'h5'


    msh.genereate_line_mesh(0.0,rpam.parameters['L'],
    len(bottom_edge_vertices)-1,rpam.parameters['sub_mesh_2_id'], rpam.parameters['vertex_sub_mesh_2_l_id'],
    rpam.parameters['vertex_sub_mesh_2_r_id'],output_directory=sub_mesh_2_output_directory,metadata=sub_mesh_2_metadata,
    coordinates=bottom_edge_vertices
)






    print("...done!")
    
else:
    print("Error: Not enough vertices found on top edge")

