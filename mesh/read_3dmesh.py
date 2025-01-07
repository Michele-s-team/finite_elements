'''
run with
clear; clear; python3 read_3dmesh.py [path where to find the mesh]
example:
clear; clear; python3 read_3dmesh.py /home/fenics/shared/mesh/solution
'''

import h5py
from mshr import *
from mshr import *
import numpy as np
from dolfin import *
import meshio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
args = parser.parse_args()

#CHANGE PARAMETERS HERE
r = 1
c_r = [0, 0, 0]
#CHANGE PARAMETERS HERE

#read the mesh of the ball
ball_mesh = Mesh()
xdmf = XDMFFile(ball_mesh.mpi_comm(), (args.input_directory) + "/tetrahedron_mesh.xdmf")
xdmf.read(ball_mesh)

#extract the boundary of the ball (spere) and write it in a new mesh `sphere`
dim=3
bdim = dim-1
sphere_mesh = BoundaryMesh( ball_mesh, "exterior" )
print("Dimension of boundary_mesh = ", sphere_mesh.geometry().dim() )


#test the mesh `sphere` by integrating a function over it
#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegral(UserExpression):
    def eval(self, values, x):
        values[0] = (x[2]/sqrt(x[0]**2 + x[1]**2 + x[2]**2))**2 * (1.0/(1.0 + (x[1]/x[0])**2))**2
    def value_shape(self):
        return (1,)


#read the tetrahedra
mvc = MeshValueCollection("size_t", sphere_mesh, sphere_mesh.topology().dim() )
cf = cpp.mesh.MeshFunctionSizet( sphere_mesh, mvc )
dx_custom = Measure("dx", domain=sphere_mesh, subdomain_data=cf )    # Line measure

Q = FunctionSpace( sphere_mesh, 'P', 1 )
f_test = Function( Q )

# f_test_ds is a scalar function defined on the mesh `sphere`
f_test.interpolate( FunctionTestIntegral( element=Q.ufl_element() ) )

#print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
print(f"Volume = {assemble( f_test * dx_custom )}, should be 1.5708" )
