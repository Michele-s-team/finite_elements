'''
Notation:
- sub_mesh: either of the parts of the total mesh

- sf_sub_mesh: a list of map functions, where sf_sub_mesh[i] is the map function for the triangles of the i-th sub_mesh
- mf_sub_mesh: a list of map functions, where mf_sub_mesh[i] is the map function for the lines of the i-th sub_mesh
'''

from fenics import *
import numpy as np

import input_output as io
import load_mesh as lmsh
import mesh as msh
import runtime_arguments as rarg

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")

print(f'**** DIMENSION *** = {lmsh.mesh.topology().dim()}')

# read the lines
cf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), io.add_trailing_slash(rarg.args.input_directory) + "line_mesh.h5", "cf")
# read the vertices
vf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, io.add_trailing_slash(rarg.args.input_directory) + "vertex_mesh.h5", "vf")

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

dx = Measure("dx", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=parameters['line_id'])
ds_l = Measure("ds", domain=lmsh.mesh, subdomain_data=vf, subdomain_id=parameters['vertex_l_id'])
ds_r = Measure("ds", domain=lmsh.mesh, subdomain_data=vf, subdomain_id=parameters['vertex_r_id'])
ds = Measure("ds", domain=lmsh.mesh)

import check_mesh_tags_square_no_circle_line

print(f'Module {__file__} called {check_mesh_tags_square_no_circle_line.__file__}', flush=True)

boundary = 'on_boundary'
boundary_l = f'near(x[0], {parameters["x_l"]})'
boundary_r = f'near(x[0], {parameters["x_r"]})'
