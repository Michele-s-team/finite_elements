'''
This code reads the 3d mesh generated from generate_box_mesh.py and it creates dvs and dss from labelled components of the mesh
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
L = [3, 2, 1]

volume_id = 1
boundary_le_id = 2
boundary_ri_id = 3
boundary_to_id = 4
boundary_bo_id = 5
boundary_fr_id = 6
boundary_ba_id = 7
# CHANGE PARAMETERS HERE

# read the tetrahedra
cf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), (rarg.args.input_directory) + "/tetrahedron_mesh.xdmf")
# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, (rarg.args.input_directory) + "/triangle_mesh.xdmf")

boundary_mesh = BoundaryMesh(lmsh.mesh, "exterior")
with XDMFFile("solution/boundary_mesh.xdmf") as xdmf:
    xdmf.write(boundary_mesh)

dx = Measure("dx", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=volume_id)  # volume measure

ds_le = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=2)
ds_ri = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=3)
ds_to = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=4)
ds_bo = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=5)
ds_fr = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=6)
ds_ba = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=7)

ds_leri = ds_le + ds_ri
ds_tobo = ds_to + ds_bo
ds_frba = ds_fr + ds_ba

ds = ds_leri + ds_tobo + ds_frba

# dS_custom = Measure("dS", domain=lmsh.mesh, subdomain_data=sf)    # Point measure for points in the mesh


import check_mesh_tags_box

print(f'Module {__file__} called {check_mesh_tags_box.__file__}', flush=True)

# Define boundaries
boundary = 'on_boundary'
boundary_le = f'near(x[0], 0)'
boundary_ri = f'near(x[0], {L[0]})'
boundary_to = f'near(x[1], {L[1]})'
boundary_bo = f'near(x[1], 0)'
boundary_fr = f'near(x[2], {L[2]})'
boundary_ba = f'near(x[2], 0)'

boundary_leri = f'near(x[0], 0) || near(x[0], {L[0]})'
boundary_tobo = f'near(x[1], 0) || near(x[1], {L[1]})'
boundary_frba = f'near(x[2], 0) || near(x[2], {L[2]})'

