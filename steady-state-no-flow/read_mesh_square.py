from fenics import *
from mshr import *
import argparse
import numpy as np

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
r_mesh = mesh.hmin()


#CHANGE PARAMETERS HERE
L = 0.5
h = L
r = 0.05
c_r = [L/2.0, h/2.0]
#CHANGE PARAMETERS HERE




# test for surface elements
ds_l = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=2 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=3 )
ds_t = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=4 )
ds_b = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=5 )
ds_circle = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=6 )
# ds_lr = ds_l + ds_r
# ds_tb = ds_t + ds_b

#a function space used solely to define f_test_ds
Q_test = FunctionSpace( bgeo.mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_test )

#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegralsds(UserExpression):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = cos(my_norm(np.subtract(x, c_test)) - r_test)**2.0
    def value_shape(self):
        return (1,)

f_test_ds.interpolate( FunctionTestIntegralsds( element=Q_test.ufl_element() ) )

# here I integrate \int ds 1 over the circle and store the result of the integral as a double in inner_circumference
integral_l = assemble( f_test_ds * ds_l )
integral_r = assemble( f_test_ds * ds_r )
integral_t = assemble( f_test_ds * ds_t )
integral_b = assemble( f_test_ds * ds_b )
integral_circle = assemble( f_test_ds * ds_circle )

# print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
print( "Integral l = ", integral_l, " exact value = 0.462517" )
print( "Integral r = ", integral_r, " exact value = 0.47113" )
print( "Integral t = ", integral_t, " exact value = 0.498266" )
print( "Integral b = ", integral_b, " exact value = 0.413016" )
print( "Integral circle = ", integral_circle, " exact value = 0.304937" )

# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_l  = 'near(x[0], 0.0)'
boundary_r  = 'near(x[0], 0.5)'
boundary_lr  = 'near(x[0], 0) || near(x[0], 0.5)'
boundary_tb  = 'near(x[1], 0) || near(x[1], 0.5)'
boundary_square = 'on_boundary && sqrt(pow(x[0] - 0.5/2.0, 2) + pow(x[1] - 0.5/2.0, 2)) > 2 * 0.05'
boundary_circle = 'on_boundary && sqrt(pow(x[0] - 0.5/2.0, 2) + pow(x[1] - 0.5/2.0, 2)) < 2 * 0.05'
#CHANGE PARAMETERS HERE