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
        values[0] = cos(x[0]**3)
    def value_shape(self):
        return (1,)

dv_custom = Measure("dx", domain=mesh, subdomain_data=cf)    # Volume measure


Q = FunctionSpace( mesh, 'P', 1 )


# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q )
f_test_ds.interpolate( FunctionTestIntegral( element=Q.ufl_element() ))


#print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
print(f"Integral of a function =  {assemble( f_test_ds * dv_custom )}")
print(f"Volume = {assemble(Constant(1.0)*dv_custom)}")
