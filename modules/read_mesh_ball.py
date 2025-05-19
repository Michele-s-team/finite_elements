'''
This code reads the 3d mesh generated from generate_ball_mesh.py and it creates dvs and dss from labelled components of the mesh
'''

import dolfin
from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import load_mesh as lmsh
import mesh as msh
import runtime_arguments as rarg

# CHANGE PARAMETERS HERE
r = 1
c_r = [0, 0, 0]

volume_id = 1
surface_id = 2
# CHANGE PARAMETERS HERE

# read the tetrahedra
cf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), (rarg.args.input_directory) + "/tetrahedron_mesh.xdmf")
# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, (rarg.args.input_directory) + "/triangle_mesh.xdmf")

boundary_mesh = BoundaryMesh(lmsh.mesh, "exterior")
with XDMFFile("solution/boundary_mesh.xdmf") as xdmf:
    xdmf.write(boundary_mesh)

'''
#read the lines
mvc = MeshValueCollection("size_t", lmsh.mesh, lmsh.mesh.topology().dim())
with XDMFFile((rarg.args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(lmsh.mesh, mvc)
xdmf.close()

#read the vertices
mvc = MeshValueCollection("size_t", lmsh.mesh, lmsh.mesh.topology().dim()-1)
with XDMFFile((rarg.args.input_directory) + "/vertex_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = cpp.mesh.MeshFunctionSizet(lmsh.mesh, mvc)
xdmf.close()
'''

dx = Measure("dx", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=volume_id)  # volume measure
ds = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=surface_id)  # surface measure
# dS_custom = Measure("dS", domain=lmsh.mesh, subdomain_data=sf)    # Point measure for points in the mesh


import check_mesh_tags_ball

print(f'Module {__file__} called {check_mesh_tags_ball.__file__}', flush=True)

# Define boundaries
boundary = 'on_boundary'
