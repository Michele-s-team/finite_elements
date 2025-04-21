from fenics import *
from mshr import *

import boundary_geometry as bgeo
import runtime_arguments as rarg

parser = rarg.argparse.ArgumentParser()
parser.add_argument( "input_directory" )
parser.add_argument( "output_directory" )
parser.add_argument( "T" )
parser.add_argument( "N" )
args = parser.parse_args()

# read the triangles
mvc = MeshValueCollection( "size_t", bgeo.mesh, bgeo.mesh.topology().dim() )
with XDMFFile( (rarg.args.input_directory) + "/triangle_mesh.xdmf" ) as infile:
    infile.read( mvc, "name_to_read" )
sf = dolfin.cpp.mesh.MeshFunctionSizet( bgeo.mesh, mvc )

# read the lines
mvc = MeshValueCollection( "size_t", bgeo.mesh, bgeo.mesh.topology().dim() - 1 )
with XDMFFile( (rarg.args.input_directory) + "/line_mesh.xdmf" ) as infile:
    infile.read( mvc, "name_to_read" )
mf = dolfin.cpp.mesh.MeshFunctionSizet( bgeo.mesh, mvc )

#radius of the smallest cell in the mesh
r_mesh = bgeo.mesh.hmin()

# CHANGE PARAMETERS HERE
# r, R must be the same as in generate_mesh.py
L = 2.2
h = 0.41
r = 0.05
c_r = [0.2, 0.2]
# CHANGE PARAMETERS HERE


# test for surface elements
dx = Measure( "dx", domain=bgeo.mesh, subdomain_data=sf, subdomain_id=1 )
ds_l = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=2 )
ds_r = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=3 )
ds_t = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=4 )
ds_b = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=5 )
ds_circle = Measure( "ds", domain=bgeo.mesh, subdomain_data=mf, subdomain_id=6 )

ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds_square = ds_lr + ds_tb
ds = ds_square + ds_circle

'''
# a function space used solely to define f_test_ds
Q_test = FunctionSpace( mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_test )


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals( UserExpression ):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = cos( my_norm( np.subtract( x, c_test ) ) - r_test ) ** 2.0

    def value_shape(self):
        return (1,)


f_test_ds.interpolate( FunctionTestIntegrals( element=Q_test.ufl_element() ) )

# here I integrate \int ds 1 over the circle and store the result of the integral as a double in inner_circumference
integral_l = assemble( f_test_ds * ds_l )
integral_r = assemble( f_test_ds * ds_r )
integral_t = assemble( f_test_ds * ds_t )
integral_b = assemble( f_test_ds * ds_b )
integral_circle = assemble( f_test_ds * ds_circle )

exact_value_int_ds_l = 0.373168
numerical_value_int_ds_sphere = assemble( f_test_ds * ds_l )
print(
    f"\int_sphere f ds = {numerical_value_int_ds_sphere}, should be  {exact_value_int_ds_l}, relative error =  {abs( (numerical_value_int_ds_sphere - exact_value_int_ds_l) / exact_value_int_ds_l ):e}" )

exact_value_int_ds_r = 0.00227783
numerical_value_int_ds_sphere = assemble( f_test_ds * ds_r )
print(
    f"\int_sphere f ds = {numerical_value_int_ds_sphere}, should be  {exact_value_int_ds_r}, relative error =  {abs( (numerical_value_int_ds_sphere - exact_value_int_ds_r) / exact_value_int_ds_r ):e}" )

exact_value_int_ds_t = 1.36562
numerical_value_int_ds_sphere = assemble( f_test_ds * ds_t )
print(
    f"\int_sphere f ds = {numerical_value_int_ds_sphere}, should be  {exact_value_int_ds_t}, relative error =  {abs( (numerical_value_int_ds_sphere - exact_value_int_ds_t) / exact_value_int_ds_t ):e}" )

exact_value_int_ds_b = 1.02837
numerical_value_int_ds_sphere = assemble( f_test_ds * ds_b )
print(
    f"\int_sphere f ds = {numerical_value_int_ds_sphere}, should be  {exact_value_int_ds_b}, relative error =  {abs( (numerical_value_int_ds_sphere - exact_value_int_ds_b) / exact_value_int_ds_b ):e}" )

exact_value_int_ds_circle = 0.298174
numerical_value_int_ds_sphere = assemble( f_test_ds * ds_circle )
print(
    f"\int_sphere f ds = {numerical_value_int_ds_sphere}, should be  {exact_value_int_ds_circle}, relative error =  {abs( (numerical_value_int_ds_sphere - exact_value_int_ds_circle) / exact_value_int_ds_circle ):e}" )
'''

import check_mesh_tags_bc_obstacle

# Define boundaries and obstacle
# CHANGE PARAMETERS HERE
inflow = 'near(x[0], 0)'
outflow = 'near(x[0], 2.2)'
walls = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
# CHANGE PARAMETERS HERE
