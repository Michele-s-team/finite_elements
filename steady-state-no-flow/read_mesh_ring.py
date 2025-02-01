from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import ufl as ufl

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
r = 1.0
R = 2.0
c_r = [0, 0]
c_R = [0, 0]

tol = 1E-3
#CHANGE PARAMETERS HERE



# test for surface elements
dx = Measure( "dx", domain=bgeo.mesh, subdomain_data=sf, subdomain_id=1 )
ds_r = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=2 )
ds_R = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=3 )
ds = ds_r + ds_R


#a function space used solely to define f_test_ds
Q_test = FunctionSpace( bgeo.mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_test )

#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegralsds(UserExpression):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test)**2.0
    def value_shape(self):
        return (1,)

f_test_ds.interpolate( FunctionTestIntegralsds( element=Q_test.ufl_element() ) )

msh.test_mesh_integral(2.90212, f_test_ds, dx, '\int f dx')
msh.test_mesh_integral(2.77595, f_test_ds, ds_r, '\int f ds_r')
msh.test_mesh_integral(3.67175, f_test_ds, ds_R, '\int f ds_R')

# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_r = 'on_boundary && sqrt(pow(x[0], 2) + pow(x[1], 2)) < (1.0 + 2.0)/2.0'
boundary_R = 'on_boundary && sqrt(pow(x[0], 2) + pow(x[1], 2)) > (1.0 + 2.0)/2.0'
#CHANGE PARAMETERS HERE