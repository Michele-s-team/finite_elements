from fenics import *

import calculus as calc
import input_output as io
import load_mesh as lmsh
import mesh as msh
import runtime_arguments as rarg

# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, 2, rarg.args.input_directory + "/triangle_mesh.xdmf")
# read the lines
mf = msh.read_mesh_components(lmsh.mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")



#
submesh_out = SubMesh(lmsh.mesh, sf, parameters["surface_out_id"])



#
# test for surface elements
dx_in = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=parameters["surface_in_id"])
dx_out = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=parameters["surface_out_id"])

# line elements for out square
ds_out_l = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_out_l_id"])
ds_out_r = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_out_r_id"])
ds_out_t = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_out_t_id"])
ds_out_b = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_out_b_id"])



# line elements for in square
ds_in_l = Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_in_l_id"])
ds_in_r = Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_in_r_id"])
ds_in_t = Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_in_t_id"])
ds_in_b = Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_in_b_id"])

ds_out_lr = ds_out_l + ds_out_r
ds_out_tb = ds_out_t + ds_out_b

ds_in_lr = ds_in_l + ds_in_r
ds_in_tb = ds_in_t + ds_in_b

ds_out = ds_out_lr + ds_out_tb
ds_in = ds_in_lr + ds_in_tb

ds = ds_in + ds_out

import check_mesh_tags_square_square

print(f'Module {__file__} called {check_mesh_tags_square_square.__file__}', flush=True)


# Define boundaries
boundary = 'on_boundary'

# out boundaries
boundary_out_l = f'near(x[0], {0})'
boundary_out_r = f'near(x[0], {parameters["L"]})'
boundary_out_t = f'near(x[1], {parameters["h"]})'
boundary_out_b = f'near(x[1], {0})'
boundary_out_lr = f'({boundary_out_l}) || ({boundary_out_r})'
boundary_out_tb = f'({boundary_out_t}) || ({boundary_out_b})'
boundary_out = f'({boundary_out_lr}) || ({boundary_out_tb})'

# in boundaries
boundary_in_l = f'near(x[0], {parameters["p"][0]})'
boundary_in_r = f'near(x[0], {parameters["p"][0] + parameters["L_in"]})'
boundary_in_t = f'near(x[1], {parameters["p"][1] + parameters["h_in"]})'
boundary_in_b = f'near(x[1], {parameters["p"][1]})'
boundary_in_lr = f'({boundary_in_l}) || ({boundary_in_r})'
boundary_in_tb = f'({boundary_in_t}) || ({boundary_in_b})'
boundary_in = f'({boundary_in_lr}) || ({boundary_in_tb})'


