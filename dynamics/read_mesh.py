from fenics import *
from mshr import *

import boundary_geometry as bgeo
import calc
import runtime_arguments as rarg


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
L = 2.2
h = 0.41
r = 0.05
c_r = [0.2, h/2.0]
#CHANGE PARAMETERS HERE

#test for surface elements
dx = Measure( "dx", domain=bgeo.mesh, subdomain_data=sf, subdomain_id=1 )
ds_l = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=2 )
ds_r = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=3 )
ds_t = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=4 )
ds_b = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=5 )
ds_circle = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=6 )
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds_square = ds_lr + ds_tb
ds = ds_square + ds_circle

import check_mesh_tags

# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_l  = f'near(x[0], 0.0)'
boundary_r  = f'near(x[0], {L})'
boundary_lr  = f'near(x[0], 0) || near(x[0], {L})'
boundary_tb  = f'near(x[1], 0) || near(x[1], {h})'
boundary_square = f'on_boundary && sqrt(pow(x[0] - {c_r[0]}, 2) + pow(x[1] - {c_r[1]}, 2)) > {(r + calc.min_dist_c_r_rectangle(L, h, c_r))/2}'
boundary_circle = f'on_boundary && sqrt(pow(x[0] - {c_r[0]}, 2) + pow(x[1] - {c_r[1]}, 2)) < {(r + calc.min_dist_c_r_rectangle(L, h, c_r))/2}'
#CHANGE PARAMETERS HERE