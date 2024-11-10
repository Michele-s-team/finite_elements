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
xdmf = XDMFFile(mesh.mpi_comm(), "tetra_mesh.xdmf")
xdmf.read(mesh)

#read the tetrahedra
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile("tetra_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#read the triangles
# mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
# with XDMFFile("triangle_mesh.xdmf") as infile:
#     infile.read(mvc, "name_to_read")
# sf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
# xdmf.close()

# mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-2)
# with XDMFFile("line_mesh.xdmf") as infile:
#     infile.read(mvc, "name_to_read")
# mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)


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