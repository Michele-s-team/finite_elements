'''
This code reads the 2d mesh generated from generate_2dmesh_ring.py and it creates dvs and dss from labelled components of the mesh


run with
clear; clear; python3 read_mesh.py [path where to find the mesh]
example:
clear; clear; python3 read_mesh.py /home/fenics/shared/generate-mesh/2d/symmetric-ring-2/solution
'''

from fenics import *
from mshr import *
from dolfin import *
from dolfin import *
import argparse
import numpy as np

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append( module_path )

import geometry as geo
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument( "input_directory" )
args = parser.parse_args()

# CHANGE PARAMETERS HERE
c_r = [0, 0, 0]
c_R = [0, 0, 0]
r = 1
R = 2
# CHANGE PARAMETERS HERE

# read the mesh
mesh = Mesh()
xdmf = XDMFFile( mesh.mpi_comm(), (args.input_directory) + "/triangle_mesh.xdmf" )
xdmf.read( mesh )
xdmf.close()

print( f"Mesh dimension = {mesh.topology().dim()}" )

# read the triangles
mvc = MeshValueCollection( "size_t", mesh, mesh.topology().dim() )
with XDMFFile( (args.input_directory) + "/triangle_mesh.xdmf" ) as infile:
    infile.read( mvc, "name_to_read" )
vf = cpp.mesh.MeshFunctionSizet( mesh, mvc )
# print("Volume measure tags:", vf.array())

# read the lines
mvc = MeshValueCollection( "size_t", mesh, mesh.topology().dim() - 1 )
with XDMFFile( (args.input_directory) + "/line_mesh.xdmf" ) as infile:
    infile.read( mvc, "name_to_read" )
sf = cpp.mesh.MeshFunctionSizet( mesh, mvc )


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegral( UserExpression ):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345

        values[0] = np.cos( geo.my_norm( np.subtract( x, c_test ) ) - r_test ) ** 2.0

    def value_shape(self):
        return (1,)


dx = Measure( "dx", domain=mesh, subdomain_data=vf, subdomain_id=1 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=2 )
ds_R = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=3 )

Q = FunctionSpace( mesh, 'P', 1 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q )
f_test_ds.interpolate( FunctionTestIntegral( element=Q.ufl_element() ) )

# print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
msh.test_mesh_integral( 2.9021223108952894, f_test_ds, dx, '\int f dx' )
msh.test_mesh_integral( 2.7759459256115657, f_test_ds, ds_r, 'int f ds_r' )
msh.test_mesh_integral( 3.6717505977470717, f_test_ds, ds_R, 'int f ds_R' )
