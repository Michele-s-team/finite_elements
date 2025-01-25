import numpy as np
import ufl as ufl


from fenics import *
from dolfin import *
from mshr import *
import runtime_arguments as rarg
import geometry as geo


#read the mesh
mesh = Mesh()
xdmf = XDMFFile(mesh.mpi_comm(), (rarg.args.input_directory) + "/triangle_mesh.xdmf")
xdmf.read(mesh)

i, j, k, l = ufl.indices(4)

#Nt^i_notes on \partisal \Omega_O
def Nt_circle(omega):
    N3d = as_tensor([facet_normal[0], facet_normal[1], 0.0])
    return as_tensor(geo.g_c(omega)[i, j] * N3d[k] * geo.e(omega)[j, k], (i))

#N_n_notes on \partial \Omega_O
def Nn_circle(omega):
    N3d = as_tensor([facet_normal[0], facet_normal[1], 0.0])
    return (N3d[i] * (normal(omega))[i])


#read the triangles
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile((rarg.args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#read the lines
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile((rarg.args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#radius of the smallest cell in the mesh
r_mesh = mesh.hmin()

#CHANGE PARAMETERS HERE
r = 1.0
R = 2.0
c_r = [0, 0]
c_R = [0, 0]

tol = 1E-3
#CHANGE PARAMETERS HERE


# norm of vector x
def my_norm(x):
    return (sqrt(np.dot(x, x)))

#this is the facet normal vector, which cannot be plotted as a field. It is not a vector in the tangent bundle of \Omega
facet_normal = FacetNormal( mesh )

# test for surface elements
dx = Measure( "dx", domain=mesh, subdomain_data=sf, subdomain_id=1 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=2 )
ds_R = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=3 )


#a function space used solely to define f_test_ds
Q_test = FunctionSpace( mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_test )

#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegralsds(UserExpression):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = cos(my_norm(np.subtract(x, c_test)) - r_test)**2.0
    def value_shape(self):
        return (1,)

f_test_ds.interpolate( FunctionTestIntegralsds( element=Q_test.ufl_element() ) )

#print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
numerical_value_int_dx = assemble( f_test_ds * dx )
exact_value_int_dx = 2.90212
print(f"\int_box_minus_ball f dx = {numerical_value_int_dx}, should be  {exact_value_int_dx}, relative error =  {abs( (numerical_value_int_dx - exact_value_int_dx) / exact_value_int_dx ):e}" )

exact_value_int_ds_r = 2.77595
numerical_value_int_ds_r = assemble( f_test_ds * ds_r )
print(f"\int_sphere f ds = {numerical_value_int_ds_r}, should be  {exact_value_int_ds_r}, relative error =  {abs( (numerical_value_int_ds_r - exact_value_int_ds_r) / exact_value_int_ds_r ):e}" )

exact_value_int_ds_R = 3.67175
numerical_value_int_ds_R = assemble( f_test_ds * ds_R )
print(f"\int_sphere f ds = {numerical_value_int_ds_R}, should be  {exact_value_int_ds_R}, relative error =  {abs( (numerical_value_int_ds_R - exact_value_int_ds_R) / exact_value_int_ds_R ):e}" )



# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_r = 'on_boundary && sqrt(pow(x[0], 2) + pow(x[1], 2)) < (1.0 + 2.0)/2.0'
boundary_R = 'on_boundary && sqrt(pow(x[0], 2) + pow(x[1], 2)) > (1.0 + 2.0)/2.0'
#CHANGE PARAMETERS HERE