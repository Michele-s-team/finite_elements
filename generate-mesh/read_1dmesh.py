'''
This code reads the 1d mesh generated from generate_1dmesh.py and it creates dvs and dss from labelled components of the mesh

run with
clear; clear; python3 read_1dmesh.py [path where to find the mesh]
example:
clear; clear; python3 read_1dmesh.py /home/fenics/shared/generate-mesh/solution
'''
from fenics import *
from mshr import *
import numpy as np
import argparse
from dolfin import *
import sys

#add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh as msh


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
xdmf = XDMFFile(mesh.mpi_comm(), (args.input_directory) + "/line_mesh.xdmf")
xdmf.read(mesh)

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

#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegral(UserExpression):
    def eval(self, values, x):
        values[0] = (np.cos(3+x[0]))**2
    def value_shape(self):
        return (1,)

dv_custom = Measure("dx", domain=mesh, subdomain_data=cf)    # Line measure
ds_custom = Measure("ds", domain=mesh, subdomain_data=sf)    # Point measure for points at the edges of the mesh
dS_custom = Measure("dS", domain=mesh, subdomain_data=sf)    # Point measure for points in the mesh

Q = FunctionSpace( mesh, 'P', 1 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q )
f_test_ds.interpolate( FunctionTestIntegral( element=Q.ufl_element() ))

#print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
msh.test_mesh_integral(1.0, Constant(1.0), dv_custom, 'Volume')
msh.test_mesh_integral(0.817193, f_test_ds, dv_custom, '\int f_{dv_custom}')
msh.test_mesh_integral(0.386545, f_test_ds, dv_custom(1), '\int f_{line #1}')
msh.test_mesh_integral(0.430648, f_test_ds, dv_custom(2), '\int f_{line #2}')

#this computes \sum_{i \in vertices in ds_custom} f_test_ds (i-th vertex in ds_custom)
msh.test_mesh_integral(0.980085, f_test_ds, ds_custom(3), '\int f_{point_l}')
msh.test_mesh_integral(0.42725, f_test_ds, ds_custom(4), '\int f_{point_r}')
msh.test_mesh_integral(0.93826, f_test_ds,  dS_custom(5), '\int f_{point_in}')
