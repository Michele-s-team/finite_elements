from __future__ import print_function
from fenics import *
from mshr import *
from fenics import *
from mshr import *
import numpy as np
import ufl as ufl
import argparse
from dolfin import *


parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
args = parser.parse_args()

#CHANGE PARAMETERS HERE
L = 1
h = 1
r = 1
c_r = [0, 0]
#CHANGE PARAMETERS HERE

#read the mesh
mesh = Mesh()
xdmf = XDMFFile(mesh.mpi_comm(), "line_mesh.xdmf")
xdmf.read(mesh)

#read the tetrahedra
# mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
# with XDMFFile("tetra_mesh.xdmf") as infile:
#     infile.read(mvc, "name_to_read")
# cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
# xdmf.close()

#read the lines
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile("line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#read the points
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile("line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegral(UserExpression):
    def eval(self, values, x):
        values[0] = x[0]
    def value_shape(self):
        return (1,)

dv_custom = Measure("dx", domain=mesh, subdomain_data=cf)    # Volume measure


# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_z_n )
f_test_ds.interpolate( FunctionTestIntegralsds( element=Q_z_n.ufl_element() ))

#here I integrate \int ds 1 over the circle and store the result of the integral as a double in inner_circumference
integral_l = assemble(f_test_ds*ds_l)
integral_r = assemble(f_test_ds*ds_r)
integral_t = assemble(f_test_ds*ds_t)
integral_b = assemble(f_test_ds*ds_b)
integral_circle = assemble(f_test_ds*ds_circle)

#print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
print("Integral l = ", integral_l, " exact value = 0.373169")
print("Integral r = ", integral_r, " exact value = 0.00227783")
print("Integral t = ", integral_t, " exact value = 1.36562")
print("Integral b = ", integral_b, " exact value = 1.02837")
print("Integral circle = ", integral_circle, " exact value = 0.205204")

# Should be 2.5
print(f"Volume = {assemble(Constant(1.0)*dv_custom)}")

'''
# Define function spaces
#finite elements for sigma .... omega
P_v_bar = VectorElement( 'P', triangle, 2 )
P_w_bar = FiniteElement( 'P', triangle, 1 )
P_phi = FiniteElement('P', triangle, 1)
P_v_n = VectorElement( 'P', triangle, 2 )
P_w_n = FiniteElement( 'P', triangle, 1 )
P_omega_n = VectorElement( 'P', triangle, 3 )
P_z_n = FiniteElement( 'P', triangle, 1 )

element = MixedElement( [P_v_bar, P_w_bar, P_phi, P_v_n, P_w_n, P_omega_n, P_z_n] )
#total function space
Q = FunctionSpace(mesh, element)
#function spaces for vbar .... zn
Q_v_bar = Q.sub(0).collapse()
Q_w_bar = Q.sub(1).collapse()
Q_phi = Q.sub(2).collapse()
Q_v_n = Q.sub(3).collapse()
Q_w_n = Q.sub(4).collapse()
Q_omega_n = Q.sub(5).collapse()
Q_z_n= Q.sub(6).collapse()


#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegralsds(UserExpression):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = cos(my_norm(np.subtract(x, c_test)) - r_test)**2.0
    def value_shape(self):
        return (1,)


# norm of vector x
def my_norm(x):
    return (sqrt(np.dot(x, x)))


#read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`
mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)



'''