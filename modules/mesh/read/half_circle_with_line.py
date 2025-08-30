from fenics import *

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg


# read the triangles
vf = msh.read_mesh_components(lmsh.mesh, 2, rarg.args.input_directory + "/triangle_mesh.xdmf")
# read the lines
cf = msh.read_mesh_components(lmsh.mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")
# read the vertices
sf = msh.read_mesh_components(lmsh.mesh, 0, rarg.args.input_directory + "/vertex_mesh.xdmf")

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")


# CHANGE PARAMETERS HERE
# r = 1
c_r = [0, 0]
c_1 = [parameters["r"], 0]
c_2 = [-parameters["r"], 0]
# c_3 = [r / 2, -r / 8]
# c_4 = [-r / 2, -r / 8]
#
# p_1_id = 1
# p_2_id = 2
# p_3_id = 6
# p_4_id = 7
# line_12_id = 3
# arc_21_id = 4
# surface_id = 5
# line_34_id = 8
# CHANGE PARAMETERS HERE


dx = Measure("dx", domain=lmsh.mesh, subdomain_data=vf, subdomain_id=parameters["surface_id"])
ds_line = Measure("ds", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=parameters["line_12_id"])
ds_line_in = Measure("dS", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=parameters["line_34_id"])
ds_arc = Measure("ds", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=parameters["arc_21_id"])
dp_line_in_start = Measure("dP", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=parameters["p_1_id"])
dp_line_in_end = Measure("dP", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=parameters["p_2_id"])

ds = ds_line + ds_arc

import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.half_circle_with_line')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

# CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_line  = 'near(x[1], 0.0)'
boundary_arc = f'on_boundary && ((x[1] < 0.0) || (near(x[0], {c_1[0]}) || near(x[0], {c_2[0]})))'
# CHANGE PARAMETERS HERE
