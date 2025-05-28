from fenics import *
import dolfin

import calculus as calc
import load_mesh as lmsh
import mesh as msh
import runtime_arguments as rarg

# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), rarg.args.input_directory + "/triangle_mesh.xdmf")

# read the lines
mf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, rarg.args.input_directory + "/line_mesh.xdmf")

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

# CHANGE PARAMETERS HERE
L = 1
h = 1
# ellipse center
c = [L / 2, h / 2, 0]
# ellipse semi-major axis
a = 0.2
# ellipse semi-minor axis
b = 0.1
# rotation angle of the ellipse with respect to the x axis: the ellipse will be rotated about its left focal point
theta = np.pi / 4
# CHANGE PARAMETERS HERE


# test for surface elements
dx = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=1)
ds_l = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=2)
ds_r = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=3)
ds_t = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=4)
ds_b = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=5)
ds_ellipse = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=6)
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds_square = ds_lr + ds_tb
ds = ds_square + ds_ellipse

import check_mesh_tags_square_ellipse

print(f'Module {__file__} called {check_mesh_tags_square_ellipse.__file__}', flush=True)

msh.check_mesh_symmetry(lmsh.mesh, c)

# Define boundaries and obstacle
# CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_l = f'near(x[0], 0.0)'
boundary_r = f'near(x[0], {L})'
boundary_lr = f'near(x[0], 0) || near(x[0], {L})'
boundary_tb = f'near(x[1], 0) || near(x[1], {h})'
boundary_square = f'on_boundary && (near(x[0], 0) || near(x[0], {L}) || near(x[1], 0) || near(x[1], {h}))'
boundary_ellipse = f'on_boundary && (!near(x[0], 0))  && (!near(x[0], {L})) && (!near(x[1], 0)) && (!near(x[1], {h}))'
# CHANGE PARAMETERS HERE
