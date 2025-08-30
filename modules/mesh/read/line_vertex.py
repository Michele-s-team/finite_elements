from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg


# read the lines
cf = msh.read_mesh_components(lmsh.mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")
# read the vertices
sf = msh.read_mesh_components(lmsh.mesh, 0, rarg.args.input_directory + "/vertex_mesh.xdmf")

#radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")


dx = Measure("dx", domain=lmsh.mesh, subdomain_data=cf)  # Line measure
dp_boundary = Measure("ds", domain=lmsh.mesh, subdomain_data=sf)  # Point measure for points at the edges of the mesh
dp_bulk = Measure("dS", domain=lmsh.mesh, subdomain_data=sf)  # Point measure for points in the mesh

import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.line_vertex')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

