'''
This code reads the 3d mesh generated from generate_3dmesh.py and it creates dvs and dss from labelled components of the mesh


run with
clear; clear; python3 read_3dmesh.py [path where to find the mesh]
example:
clear; clear; python3 read_3dmesh.py /home/fenics/shared/mesh/solution
'''


from __future__ import print_function
from fenics import *
from mshr import *
from fenics import *
from mshr import *
import numpy as np
import argparse
from dolfin import *
import meshio



parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
args = parser.parse_args()

#CHANGE PARAMETERS HERE
r = 1
c_r = [0, 0]
#CHANGE PARAMETERS HERE

#read the mesh
mesh = Mesh()
xdmf = XDMFFile(mesh.mpi_comm(), (args.input_directory) + "/tetrahedron_mesh.xdmf")
xdmf.read(mesh)



#read the tetrahedra
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile((args.input_directory) + "/tetrahedron_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#read the triangles
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

boundary_mesh = BoundaryMesh(mesh, "exterior")
with XDMFFile("solution/boundary_mesh.xdmf") as xdmf:
    xdmf.write(boundary_mesh)

'''
#read the lines
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#read the vertices
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile((args.input_directory) + "/vertex_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()
'''

#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegral(UserExpression):
    def eval(self, values, x):
        values[0] = (np.cos(3+x[0]))**2
    def value_shape(self):
        return (1,)

dv_custom = Measure("dx", domain=mesh, subdomain_data=cf)    # Line measure
ds_custom = Measure("ds", domain=mesh, subdomain_data=sf, subdomain_id=1)    # Point measure for points at the edges of the mesh
# dS_custom = Measure("dS", domain=mesh, subdomain_data=sf)    # Point measure for points in the mesh

Q = FunctionSpace( mesh, 'P', 1 )


# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q )
f_test_ds.interpolate( FunctionTestIntegral( element=Q.ufl_element() ))


#print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
print(f"\int_ball dx f = {assemble(Constant(1.0)*dv_custom)}, should be 4.1887902047863905")
print(f"\int_sphere ds f = {assemble(Constant(1.0)*ds_custom)}, should be 4.1887902047863905")
# print(f"Integral over the whole domain =  {assemble( f_test_ds * dv_custom )}", " should be 0.817193")
# print(f"Integral over line #1 =  {assemble( f_test_ds * dv_custom(1) )}", "should be 0.386545")
# print(f"Integral over line #2 =  {assemble( f_test_ds * dv_custom(2) )}", "should be 0.430648")
#
# #this computes \sum_{i \in vertices in ds_custom} f_test_ds (i-th vertex in ds_custom)
# print(f"Integral over point_l  =  {assemble( f_test_ds * ds_custom(3) )} should be 0.980085")
# print(f"Integral over point_r =  {assemble( f_test_ds * ds_custom(4) )} should be 0.42725")
# print(f"Integral over point_in =  {assemble( f_test_ds * dS_custom(5) )} should be 0.93826")
