from fenics import *

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



# test for surface elements
dx = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=1)
ds = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=2)

import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.disk')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

# CHANGE PARAMETERS HERE
boundary = 'on_boundary'
# CHANGE PARAMETERS HERE
