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
numerical_value_int_dx_box_minus_ball = assemble( f_test_ds * dx_box_minus_ball )
exact_value_int_dx_box_minus_ball = 0.472941
print(f"\int_box_minus_ball f dx = {numerical_value_int_dx_box_minus_ball}, should be  {exact_value_int_dx_box_minus_ball}, relative error =  {abs( (numerical_value_int_dx_box_minus_ball - exact_value_int_dx_box_minus_ball) / exact_value_int_dx_box_minus_ball ):e}" )

exact_value_int_ds_sphere = 0.309249
numerical_value_int_ds_sphere = assemble( f_test_ds * ds_sphere )
print(f"\int_sphere f ds = {numerical_value_int_ds_sphere}, should be  {exact_value_int_ds_sphere}, relative error =  {abs( (numerical_value_int_ds_sphere - exact_value_int_ds_sphere) / exact_value_int_ds_sphere ):e}" )


exact_value_int_ds_le = 0.505778
numerical_value_int_ds_le = assemble( f_test_ds * ds_le )
print(f"\int_l f ds = {numerical_value_int_ds_le}, should be  {exact_value_int_ds_le}, relative error =  {abs( (numerical_value_int_ds_le - exact_value_int_ds_le) / exact_value_int_ds_le ):e}" )

exact_value_int_ds_ri = 0.489414
numerical_value_int_ds_ri = assemble( f_test_ds * ds_ri )
print(f"\int_r f ds = {numerical_value_int_ds_ri}, should be  {exact_value_int_ds_ri}, relative error =  {abs( (numerical_value_int_ds_ri - exact_value_int_ds_ri) / exact_value_int_ds_ri ):e}" )

exact_value_int_ds_fr = 0.487063
numerical_value_int_ds_fr = assemble( f_test_ds * ds_fr )
print(f"\int_fr f ds = {numerical_value_int_ds_fr}, should be  {exact_value_int_ds_fr}, relative error =  {abs( (numerical_value_int_ds_fr - exact_value_int_ds_fr) / exact_value_int_ds_fr ):e}" )

exact_value_int_ds_ba = 0.519791
numerical_value_int_ds_ba = assemble( f_test_ds * ds_ba )
print(f"\int_ba f ds = {numerical_value_int_ds_ba}, should be  {exact_value_int_ds_ba}, relative error =  {abs( (numerical_value_int_ds_ba - exact_value_int_ds_ba) / exact_value_int_ds_ba ):e}" )

exact_value_int_ds_to = 0.554261
numerical_value_int_ds_to = assemble( f_test_ds * ds_to )
print(f"\int_to f ds = {numerical_value_int_ds_to}, should be  {exact_value_int_ds_to}, relative error =  {abs( (numerical_value_int_ds_to - exact_value_int_ds_to) / exact_value_int_ds_to ):e}" )

exact_value_int_ds_bo = 0.603353
numerical_value_int_ds_bo = assemble( f_test_ds * ds_bo )
print(f"\int_bo f ds = {numerical_value_int_ds_bo}, should be  {exact_value_int_ds_bo}, relative error =  {abs( (numerical_value_int_ds_bo - exact_value_int_ds_bo) / exact_value_int_ds_bo ):e}" )
