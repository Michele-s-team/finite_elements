from fenics import *
from dolfin import *
from mshr import *

import runtime_arguments as rarg
import boundary_geometry as bgeo


#read the triangles
mvc = MeshValueCollection("size_t", bgeo.mesh, bgeo.mesh.topology().dim())
with XDMFFile((rarg.args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = dolfin.cpp.mesh.MeshFunctionSizet(bgeo.mesh, mvc)

#read the lines
mvc = MeshValueCollection("size_t", bgeo.mesh, bgeo.mesh.topology().dim()-1)
with XDMFFile((rarg.args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = dolfin.cpp.mesh.MeshFunctionSizet(bgeo.mesh, mvc)

#radius of the smallest cell in the mesh
r_mesh = bgeo.mesh.hmin()


#CHANGE PARAMETERS HERE
L = 1.0
h = 1.0
#CHANGE PARAMETERS HERE

dx = Measure( "dx", domain=bgeo.mesh, subdomain_data=sf, subdomain_id=1 )
ds_l = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=2 )
ds_r = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=3 )
ds_t = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=4 )
ds_b = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=5 )
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds = ds_lr + ds_tb

import check_mesh_tags_square_no_circle

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