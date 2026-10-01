'''
This code reads the 3d mesh generated from generate_mesh.py and it creates dvs and dss from labelled components of the mesh
'''

from fenics import *
import importlib
import os
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg

# 1. read mesh components

#1.1 read the tetrahedra
cf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), os.path.join(rarg.args.input_directory, 'tetra_mesh.xdmf'))

#1.2 read the triangles
# 1.2.1 boundary triangles
sf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, os.path.join(rarg.args.input_directory, 'triangle_mesh.xdmf'))

# 1.2.2 internal triangles
sf_I = msh.read_mesh_internal_components(lmsh.mesh, cf, lmsh.parameters['surface_volume_id'], lmsh.parameters['box_volume_id'], lmsh.parameters['surface_surface_id'])


#2.  radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()


# 3. define measures

# 3.1 volume measures
dx_surface = Measure("dx", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=lmsh.parameters['surface_volume_id'])  
dx_box = Measure("dx", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=lmsh.parameters['box_volume_id'])  

dx = dx_box + dx_surface


# 3.2 surface measures

# 3.2.1 boundary surface measures

ds_le = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['boundary_le_id'])
ds_ri = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['boundary_ri_id'])
ds_to = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['boundary_to_id'])
ds_bo = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['boundary_bo_id'])
ds_fr = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['boundary_fr_id'])
ds_ba = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['boundary_ba_id'])


ds_leri = ds_le + ds_ri
ds_tobo = ds_to + ds_bo
ds_frba = ds_fr + ds_ba

ds = ds_leri + ds_tobo + ds_frba

# 3.2.2 internal surface measures

dS_surface = Measure("dS", domain=lmsh.mesh, subdomain_data=sf_I, subdomain_id=lmsh.parameters[f"surface_surface_id"])


check_mesh_module = importlib.import_module('mesh.check_tags.box_surface')
print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)
