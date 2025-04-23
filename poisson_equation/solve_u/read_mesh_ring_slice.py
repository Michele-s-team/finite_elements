from dolfin import *
from fenics import *
from mshr import *
import numpy as np

import calculus as cal
import runtime_arguments as rarg
import boundary_geometry as bgeo

# read the triangles
mvc = MeshValueCollection("size_t", bgeo.mesh, bgeo.mesh.topology().dim())
with XDMFFile((rarg.args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = dolfin.cpp.mesh.MeshFunctionSizet(bgeo.mesh, mvc)

# read the lines
mvc = MeshValueCollection("size_t", bgeo.mesh, bgeo.mesh.topology().dim() - 1)
with XDMFFile((rarg.args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = dolfin.cpp.mesh.MeshFunctionSizet(bgeo.mesh, mvc)

# radius of the smallest cell in the mesh
r_mesh = bgeo.mesh.hmin()

# CHANGE PARAMETERS HERE
r = 1
R = 2
c_r = [0, 0]
c_R = [0, 0]
N = 16
theta = 2 * np.pi / N
theta_min = 0
theta_max = 2 * 2 * np.pi / N

r_lb = np.array([r, 0])
r_lt = cal.R(theta_max).dot(r_lb)
r_rb = np.array([R, 0])
r_rt = cal.R(theta_max).dot(r_rb)

c_test = [0.3, 0.76]
r_test = 0.345

surface_id = 1
circle_r_id = 2
circle_R_id = 3
lines_tb_id = 5
line_middle_id = 4

epsilon_boundaries = 1e-3
# CHANGE PARAMETERS HERE

dx = Measure("dx", domain=bgeo.mesh, subdomain_data=sf, subdomain_id=surface_id)
ds_arc_r = Measure("ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=circle_r_id)
ds_arc_R = Measure("ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=circle_R_id)
ds_line_tb = Measure("ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=lines_tb_id)
ds_line_middle = Measure("ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=line_middle_id)
ds_arc_rR = ds_arc_r + ds_arc_R
ds = ds_arc_rR + ds_line_tb

import check_mesh_tags_ring_slice

print(f'Module {__file__} called {check_mesh_tags_ring_slice.__file__}', flush=True)

# Define boundaries and obstacle
# CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_line_t = f'near(atan2(x[1], x[0), {theta_max})'
boundary_line_b = f'near(x[0], 0.0)'
boundary_line_tb = f'near(x[0], 0.0) || near(atan2(x[1], x[0]), {theta_max})'
boundary_arc_r = f'on_boundary && && sqrt(pow(x[0] - {c_r[0]}, 2) + pow(x[1] - {c_r[1]}, 2)) < {r + epsilon_boundaries} && sqrt(pow(x[0] - {c_r[0]}, 2) + pow(x[1] - {c_r[1]}, 2)) > {r - epsilon_boundaries}'
boundary_arc_R = f'on_boundary && && sqrt(pow(x[0] - {c_R[0]}, 2) + pow(x[1] - {c_R[1]}, 2)) < {R + epsilon_boundaries} && sqrt(pow(x[0] - {c_R[0]}, 2) + pow(x[1] - {c_R[1]}, 2)) > {R - epsilon_boundaries}'
boundary_arc_rR = f'on_boundary && ((sqrt(pow(x[0] - {c_r[0]}, 2) + pow(x[1] - {c_r[1]}, 2)) < {r + epsilon_boundaries} && sqrt(pow(x[0] - {c_r[0]}, 2) + pow(x[1] - {c_r[1]}, 2)) > {r - epsilon_boundaries}) || (sqrt(pow(x[0] - {c_R[0]}, 2) + pow(x[1] - {c_R[1]}, 2)) < {R + epsilon_boundaries} && sqrt(pow(x[0] - {c_R[0]}, 2) + pow(x[1] - {c_R[1]}, 2)) > {R - epsilon_boundaries}))'
# CHANGE PARAMETERS HERE
