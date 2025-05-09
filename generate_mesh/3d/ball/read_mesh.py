'''
This code reads the 3d mesh generated from generate_mesh.py and it creates dvs and dss from labelled components of the mesh

Run with
    clear; clear; python3 read_mesh.py [path where to find the mesh]

Example:
    clear; clear; python3 read_mesh.py /home/fenics/shared/generate_mesh/3d/ball/solution
'''

import argparse
import colorama as col
import dolfin
from fenics import *
import numpy as np
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
args = parser.parse_args()

# CHANGE PARAMETERS HERE
r = 1
c_r = [0, 0]
# CHANGE PARAMETERS HERE

# read the mesh
mesh = Mesh()
xdmf = XDMFFile(mesh.mpi_comm(), (args.input_directory) + "/tetrahedron_mesh.xdmf")
xdmf.read(mesh)

# read the tetrahedra
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile((args.input_directory) + "/tetrahedron_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

# read the triangles
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
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


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegral(UserExpression):
    def eval(self, values, x):
        # values[0] = 1.0
        values[0] = (np.cos(3 * x[2] - 2 * x[1] + x[0])) ** 2

    def value_shape(self):
        return (1,)


dv_custom = Measure("dx", domain=mesh, subdomain_data=cf, subdomain_id=2)  # volume measure
ds_custom = Measure("ds", domain=mesh, subdomain_data=sf, subdomain_id=1)  # surface measure
# dS_custom = Measure("dS", domain=mesh, subdomain_data=sf)    # Point measure for points in the mesh

Q = FunctionSpace(mesh, 'P', 1)

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function(Q)
f_test_ds.interpolate(FunctionTestIntegral(element=Q.ufl_element()))

test_mesh_integral_errors = []

# print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
test_mesh_integral_errors.append(msh.test_mesh_integral(2.06773, f_test_ds, dv_custom, '\int_ball f dx'))
test_mesh_integral_errors.append(msh.test_mesh_integral(7.06579, f_test_ds, ds_custom, '\int_sphere f ds'))

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')

