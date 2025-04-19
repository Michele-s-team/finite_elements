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
r = 1.0
R = 2.0
c_r = [0, 0]
c_R = [0, 0]
#CHANGE PARAMETERS HERE



# test for surface elements
dx = Measure( "dx", domain=bgeo.mesh, subdomain_data=sf, subdomain_id=1 )
ds_r = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=2 )
ds_R = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=3 )
ds = ds_r + ds_R

import check_mesh_tags_ring

# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_r = f'on_boundary && sqrt(pow(x[0] - {c_r[0]}, 2) + pow(x[1] - {c_r[1]}, 2)) < ({r} + {R})/2.0'
boundary_R = f'on_boundary && sqrt(pow(x[0] - {c_R[0]}, 2) + pow(x[1] - {c_R[1]}, 2)) > ({r} + {R})/2.0'
#CHANGE PARAMETERS HERE