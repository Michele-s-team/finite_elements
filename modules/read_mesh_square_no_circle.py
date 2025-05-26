import dolfin
from fenics import *

import load_mesh as lmsh
import runtime_arguments as rarg

#read the triangles
mvc = MeshValueCollection("size_t", lmsh.mesh, lmsh.mesh.topology().dim())
with XDMFFile((rarg.args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = dolfin.cpp.mesh.MeshFunctionSizet(lmsh.mesh, mvc)

#read the lines
mvc = MeshValueCollection("size_t", lmsh.mesh, lmsh.mesh.topology().dim()-1)
with XDMFFile((rarg.args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = dolfin.cpp.mesh.MeshFunctionSizet(lmsh.mesh, mvc)

#radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()


#CHANGE PARAMETERS HERE
L = 1
h = 1
#CHANGE PARAMETERS HERE

dx = Measure( "dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=1 )
ds_l = Measure( "ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=2 )
ds_r = Measure( "ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=3 )
ds_t = Measure( "ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=4 )
ds_b = Measure( "ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=5 )
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds = ds_lr + ds_tb

import check_mesh_tags_square_no_circle
print(f'Module {__file__} called {check_mesh_tags_square_no_circle.__file__}', flush=True)


# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_l  = 'near(x[0], 0.0)'
boundary_r  = f'near(x[0], {L})'
boundary_t  = f'near(x[1], {h})'
boundary_b  = 'near(x[1], 0.0)'
boundary_lr  = f'near(x[0], 0) || near(x[0], {L})'
boundary_tb  = f'near(x[1], 0) || near(x[1], {h})'
#CHANGE PARAMETERS HERE