'''
This code reads the 2d mesh generated from generate_2dmesh_ring.py and it creates dvs and dss from labelled components of the mesh


run with
clear; clear; python3 read_2dmesh_ring.py [path where to find the mesh]
example:
clear; clear; python3 read_2dmesh_ring.py /home/fenics/shared/generate-mesh/solution
'''


from fenics import *
from mshr import *
from dolfin import *
from dolfin import *
import argparse
import numpy as np

import sys

#add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import geometry as geo
import mesh as msh



parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
args = parser.parse_args()

#CHANGE PARAMETERS HERE
c_r = [0, 0, 0]
c_R = [0, 0, 0]
r = 1
R = 2
#CHANGE PARAMETERS HERE

# read the mesh
mesh = Mesh()
xdmf = XDMFFile( mesh.mpi_comm(), (args.input_directory) + "/triangle_mesh.xdmf" )
xdmf.read( mesh )

print(f"Mesh dimension = {mesh.topology().dim()}")

# read the triangles
mvc = MeshValueCollection( "size_t", mesh, mesh.topology().dim() )
with XDMFFile( (args.input_directory) + "/triangle_mesh.xdmf" ) as infile:
    infile.read( mvc, "name_to_read" )
vf = cpp.mesh.MeshFunctionSizet( mesh, mvc )
xdmf.close()

#read the lines
'''
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()
'''

#read the vertices
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-2)
with XDMFFile((args.input_directory) + "/vertex_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
pf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()


#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegral(UserExpression):
    def eval(self, values, x):
        c_test = [0.3, 0.76, 0]
        r_test = 0.345

        values[0] = np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test)**2.0
    def value_shape(self):
        return (1,)



dx = Measure( "dx", domain=mesh, subdomain_data=vf, subdomain_id=6   )
# ds_r = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=1 )
# ds_R = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=2 )
# ds_line_p1_p2 = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=3    )

ds_custom = Measure("ds", domain=mesh, subdomain_data=pf)    # Point measure for points at the edges of the mesh
dS_custom = Measure("dS", domain=mesh, subdomain_data=pf)    # Point measure for points in the mesh



Q = FunctionSpace( mesh, 'P', 1 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q )
f_test_ds.interpolate( FunctionTestIntegral( element=Q.ufl_element() ))

#print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
# msh.test_mesh_integral( 0.472941, f_test_ds, dx, '\int_box_minus_ball f dx' )
# msh.test_mesh_integral(0.309249, f_test_ds, ds_sphere, '\int_sphere f ds')
# msh.test_mesh_integral(0.505778, f_test_ds, ds_le, '\int_l f ds')
# msh.test_mesh_integral(0.489414, f_test_ds, ds_ri, '\int_r f ds')
# msh.test_mesh_integral(0.487063, f_test_ds, ds_fr, '\int_fr f ds')
# msh.test_mesh_integral(0.519791, f_test_ds, ds_ba, '\int_ba f ds')
# msh.test_mesh_integral(0.554261, f_test_ds, ds_to, '\int_to f ds')

msh.test_mesh_integral(2.9021223108952894, f_test_ds, dx, '\int f dx')

