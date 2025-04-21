from fenics import *
from mshr import *

import boundary_geometry as bgeo
import runtime_arguments as rarg


parser = rarg.argparse.ArgumentParser()
parser.add_argument( "input_directory" )
parser.add_argument( "output_directory" )
parser.add_argument( "T" )
parser.add_argument( "N" )
args = parser.parse_args()


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



# CHANGE PARAMETERS HERE
L = 4.4
h = 0.41
# CHANGE PARAMETERS HERE




# test for surface elements
dx = Measure( "dx", domain=bgeo.mesh, subdomain_data=sf, subdomain_id=1 )
ds_l = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=2 )
ds_r = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=3 )
ds_t = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=4 )
ds_b = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=5 )

ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds = ds_lr + ds_tb

import check_mesh_tags_bc_no_obstacle

# Define boundaries and obstacle
# CHANGE PARAMETERS HERE
inflow = 'near(x[0], 0)'
outflow = 'near(x[0], 4.4)'
walls = 'near(x[1], 0) || near(x[1], 0.41)'
# CHANGE PARAMETERS HERE
