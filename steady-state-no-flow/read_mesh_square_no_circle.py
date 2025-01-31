from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import ufl as ufl

import runtime_arguments as rarg
import mesh as msh

#read mesh
mesh=Mesh()
with XDMFFile((rarg.args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((rarg.args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")

#radius of the smallest cell in the mesh
r_mesh = mesh.hmin()


#CHANGE PARAMETERS HERE
L = 1.0
h = 1.0

tol = 1E-3
#CHANGE PARAMETERS HERE


# norm of vector x
def my_norm(x):
    return (sqrt(np.dot(x, x)))

#this is the facet normal vector, which cannot be plotted as a field. It is not a vector in the tangent bundle of \Omega
facet_normal = FacetNormal( mesh )

# read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`
mf = dolfin.cpp.mesh.MeshFunctionSizet( mesh, mvc )

# test for surface elements
ds_l = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=2 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=3 )
ds_t = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=4 )
ds_b = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=5 )


#a function space used solely to define f_test_ds
Q_test = FunctionSpace( mesh, 'P', 2 )

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

# print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
print( "Integral l = ", integral_l, " exact value = 0.962047" )
print( "Integral r = ", integral_r, " exact value = 0.996086" )
print( "Integral t = ", integral_t, " exact value = 5.26416" )
print( "Integral b = ", integral_b, " exact value = 4.97026" )

# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_l  = 'near(x[0], 0.0)'
boundary_r  = 'near(x[0], 0.5)'
boundary_t  = 'near(x[1], 0.5)'
boundary_b  = 'near(x[1], 0.0)'
boundary_lr  = 'near(x[0], 0) || near(x[0], 0.5)'
boundary_tb  = 'near(x[1], 0) || near(x[1], 0.5)'
boundary_square = 'on_boundary'
#CHANGE PARAMETERS HERE