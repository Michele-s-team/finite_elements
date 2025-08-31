from fenics import *
import numpy as np

import calculus as cal
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg

# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, 2, rarg.args.input_directory + "/triangle_mesh.xdmf")

# read the lines
mf = msh.read_mesh_components(lmsh.mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")
theta = 2 * np.pi / parameters["N"]

theta = 2 * np.pi / parameters["N"]

r_lb = np.array([parameters["r"], 0])
r_lt = cal.R(theta).dot(r_lb)
r_rb = np.array([parameters["R"], 0])
r_rt = cal.R(theta).dot(r_rb)

c_test = [0.3, 0.76]
r_test = 0.345

surface_id = 1
circle_r_id = 2
circle_R_id = 3
line_t_id = 4
line_b_id = 5

epsilon_boundaries = 1e-3
# CHANGE PARAMETERS HERE

dx = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=surface_id)
ds_arc_r = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=circle_r_id)
ds_arc_R = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=circle_R_id)
ds_line_t = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=line_t_id)
ds_line_b = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=line_b_id)
ds_arc_rR = ds_arc_r + ds_arc_R
ds_line_tb = ds_line_t + ds_line_b
ds = ds_arc_rR + ds_line_tb

import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.ring_slice')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

# Define boundaries and obstacle
boundary = 'on_boundary'
boundary_line_t = f'near(atan2(x[1], x[0), {theta})'
boundary_line_b = f'near(x[0], 0.0)'
boundary_line_tb = f'near(x[0], 0.0) || near(atan2(x[1], x[0]), {theta})'
boundary_arc_r = f'on_boundary && && sqrt(pow(x[0] - {parameters["c_r"][0]}, 2) + pow(x[1] - {parameters["c_r"][1]}, 2)) < {parameters["r"] + epsilon_boundaries} && sqrt(pow(x[0] - {parameters["c_r"][0]}, 2) + pow(x[1] - {parameters["c_r"][1]}, 2)) > {parameters["r"] - epsilon_boundaries}'
boundary_arc_R = f'on_boundary && && sqrt(pow(x[0] - {parameters["c_R"][0]}, 2) + pow(x[1] - {parameters["c_R"][1]}, 2)) < {parameters["R"] + epsilon_boundaries} && sqrt(pow(x[0] - {parameters["c_R"][0]}, 2) + pow(x[1] - {parameters["c_R"][1]}, 2)) > {parameters["R"] - epsilon_boundaries}'
boundary_arc_rR = f'on_boundary && ((sqrt(pow(x[0] - {parameters["c_r"][0]}, 2) + pow(x[1] - {parameters["c_r"][1]}, 2)) < {parameters["r"] + epsilon_boundaries} && sqrt(pow(x[0] - {parameters["c_r"][0]}, 2) + pow(x[1] - {parameters["c_r"][1]}, 2)) > {parameters["r"] - epsilon_boundaries}) || (sqrt(pow(x[0] - {parameters["c_R"][0]}, 2) + pow(x[1] - {parameters["c_R"][1]}, 2)) < {parameters["R"] + epsilon_boundaries} && sqrt(pow(x[0] - {parameters["c_R"][0]}, 2) + pow(x[1] - {parameters["c_R"][1]}, 2)) > {parameters["R"] - epsilon_boundaries}))'
