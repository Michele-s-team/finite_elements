from fenics import *
from mshr import *
import numpy as np

import calc
import runtime_arguments as rarg
import mesh as msh
import geometry as geo
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
L = 0.5
h = L
r = 0.05
c_r = [L/2.0, h/2.0]
#CHANGE PARAMETERS HERE




# test for surface elements
dx = Measure( "dx", domain=bgeo.mesh, subdomain_data=sf, subdomain_id=1 )
ds_l = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=2 )
ds_r = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=3 )
ds_t = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=4 )
ds_b = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=5 )
ds_circle = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=6 )
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds_square = ds_lr + ds_tb

# #a function space used solely to define f_test_ds
# Q_test = FunctionSpace( bgeo.mesh, 'P', 2 )

# # f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
# f_test_ds = Function( Q_test )

# #analytical expression for a  scalar function used to test the ds
# class FunctionTestIntegralsds(UserExpression):
#     def eval(self, values, x):
#         c_test = [0.3, 0.76]
#         r_test = 0.345
#         values[0] = np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test)**2.0
#     def value_shape(self):
#         return (1,)

# f_test_ds.interpolate( FunctionTestIntegralsds( element=Q_test.ufl_element() ) )

# msh.test_mesh_integral(0.22908817224489927, f_test_ds, dx, '\int f dx')
# msh.test_mesh_integral(1.8449287777896068, f_test_ds, ds_square, '\int_square f ds')
# msh.test_mesh_integral(0.3049366444861381, f_test_ds, ds_circle, '\int_circle f ds')
# msh.test_mesh_integral(0.9336461710791771, f_test_ds, ds_lr, '\int_lr f ds')
# msh.test_mesh_integral(0.9112826067104298, f_test_ds, ds_tb, '\int_tb f ds')
#
# msh.test_mesh_integral(0.4625165259025798, f_test_ds, ds_l, '\int_l f ds')
# msh.test_mesh_integral(0.47112964517659733, f_test_ds, ds_r, '\int_r f ds')
# msh.test_mesh_integral(0.4982661696490371, f_test_ds, ds_t, '\int_t f ds')
# msh.test_mesh_integral(0.41301643706139274, f_test_ds, ds_b, '\int_b f ds')

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

