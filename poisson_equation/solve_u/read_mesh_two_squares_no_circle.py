from fenics import *
from mshr import *

import calculus as calc
import runtime_arguments as rarg
import boundary_geometry as bgeo

# read the triangles
mvc = MeshValueCollection("size_t", bgeo.mesh, bgeo.mesh.topology().dim())
with XDMFFile((rarg.args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = dolfin.cpp.mesh.MeshFunctionSizet(bgeo.mesh, mvc)

# read the lines
mvc = MeshValueCollection("size_t", bgeo.mesh, bgeo.mesh.topology().dim() - 1)
with XDMFFile((rarg.args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = dolfin.cpp.mesh.MeshFunctionSizet(bgeo.mesh, mvc)

# radius of the smallest cell in the mesh
r_mesh = bgeo.mesh.hmin()

# CHANGE PARAMETERS HERE
L = 1
h = 2
L_m = L / 3

l_surface_id = 1
r_surface_id = 2
l_line_id = 3
lb_line_id = 4
rb_line_id = 5
r_line_id = 6
tr_line_id = 7
tl_line_id = 8
m_line_id = 9
# CHANGE PARAMETERS HERE


dx_l = Measure("dx", domain=bgeo.mesh, subdomain_data=sf, subdomain_id=l_surface_id)
dx_r = Measure("dx", domain=bgeo.mesh, subdomain_data=sf, subdomain_id=r_surface_id)
ds_l = Measure("ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=l_line_id)
ds_r = Measure("ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=r_line_id)
ds_lb = Measure("ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=lb_line_id)
ds_rb = Measure("ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=rb_line_id)
ds_rt = Measure("ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=tr_line_id)
ds_lt = Measure("ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=tl_line_id)
ds_m = Measure("dS", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=m_line_id)

dx = dx_l + dx_r

ds_b = ds_lb + ds_rb
ds_t = ds_lt + ds_rt
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b

ds = ds_lr + ds_tb

import check_mesh_tags_two_squares_no_circle

print(f'Module {__file__} called {check_mesh_tags_two_squares_no_circle.__file__}', flush=True)

# Define boundaries and obstacle
# CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_l = f'near(x[0], 0.0)'
boundary_r = f'near(x[0], {L})'
boundary_lr = f'near(x[0], 0) || near(x[0], {L})'
boundary_tb = f'near(x[1], 0) || near(x[1], {h})'
# CHANGE PARAMETERS HERE
