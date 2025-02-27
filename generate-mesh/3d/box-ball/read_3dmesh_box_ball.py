'''
This code reads the 3d mesh generated from generate_3dmesh.py and it creates dvs and dss from labelled components of the mesh


run with
clear; clear; python3 read_3dmesh_box_ball.py [path where to find the mesh]
example:
clear; clear; python3 read_3dmesh_box_ball.py /home/fenics/shared/generate-mesh/solution
'''


from fenics import *
from mshr import *
from dolfin import *
import numpy as np
import argparse

import sys

#add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh as msh



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



#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegral(UserExpression):
    def eval(self, values, x):
        values[0] = (np.cos(3.0*x[2]-2.0*x[1]+x[0]))**2
    def value_shape(self):
        return (1,)

dx_box_minus_ball = Measure( "dx", domain=mesh, subdomain_data=cf, subdomain_id=8 )
ds_sphere = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=7 )
ds_le = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=1 )
ds_ri = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=2 )
ds_fr = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=3 )
ds_ba = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=4 )
ds_to = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=5 )
ds_bo = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=6 )

Q = FunctionSpace( mesh, 'P', 1 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q )
f_test_ds.interpolate( FunctionTestIntegral( element=Q.ufl_element() ))

#print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
msh.test_mesh_integral(0.472941, f_test_ds, dx_box_minus_ball, '\int_box_minus_ball f dx')
msh.test_mesh_integral(0.309249, f_test_ds, ds_sphere, '\int_sphere f ds')
msh.test_mesh_integral(0.505778, f_test_ds, ds_le, '\int_l f ds')
msh.test_mesh_integral(0.489414, f_test_ds, ds_ri, '\int_r f ds')
msh.test_mesh_integral(0.487063, f_test_ds, ds_fr, '\int_fr f ds')
msh.test_mesh_integral(0.519791, f_test_ds, ds_ba, '\int_ba f ds')
msh.test_mesh_integral(0.554261, f_test_ds, ds_to, '\int_to f ds')
msh.test_mesh_integral(0.603353, f_test_ds, ds_bo, '\int_bo f ds')
