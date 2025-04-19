from fenics import *
from dolfin import *
from mshr import *
import numpy as np

import runtime_arguments as rarg
# import mesh as msh
# import geometry as geo
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



# read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`

# test for surface elements
dx = Measure( "dx", domain=bgeo.mesh, subdomain_data=sf, subdomain_id=1 )
ds_l = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=2 )
ds_r = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=3 )
ds_t = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=4 )
ds_b = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=5 )
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds = ds_lr + ds_tb

#
# #a function space used solely to define f_test_ds
# Q_test = FunctionSpace( bgeo.mesh, 'P', 2 )
#
# # f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
# f_test_ds = Function( Q_test )
#
# #analytical expression for a  scalar function used to test the ds
# class FunctionTestIntegralsds(UserExpression):
#     def eval(self, values, x):
#         c_test = [0.3, 0.76]
#         r_test = 0.345
#         values[0] = np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test)**2.0
#     def value_shape(self):
#         return (1,)
#
# f_test_ds.interpolate( FunctionTestIntegralsds( element=Q_test.ufl_element() ) )
#
# msh.test_mesh_integral(0.9376440450770376, f_test_ds, dx, '\int f dx')
#
# msh.test_mesh_integral(3.519879453774341, f_test_ds, ds, '\int f ds')
#
# msh.test_mesh_integral(1.7676785508433541, f_test_ds, ds_lr, '\int f ds_lr')
# msh.test_mesh_integral(1.752200902930987, f_test_ds, ds_tb, '\int f ds_tb')
#
# msh.test_mesh_integral(0.9620471547152678, f_test_ds, ds_l, '\int f ds_l')
# msh.test_mesh_integral(0.8056313961280863, f_test_ds, ds_r, '\int f ds_r')
# msh.test_mesh_integral(0.9756236687066221, f_test_ds, ds_t, '\int f ds_t')
# msh.test_mesh_integral(0.7765772342243651, f_test_ds, ds_b, '\int f ds_b')

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