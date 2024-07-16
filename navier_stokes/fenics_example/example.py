'''
run with 

python3 example.py /home/fenics/shared/mesh/membrane_mesh /home/fenics/shared/navier_stokes/fenics_example/solution


'''


from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import argparse
import ufl as ufl


parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
args = parser.parse_args()

set_log_level(20)






# # Create mesh
# channel = Rectangle(Point(0, 0), Point(1.0, 1.0))
# cylinder = Circle(Point(0.2, 0.2), 0.05)
# domain = channel - cylinder
# mesh = generate_mesh(domain, 64)

L = 1.0
h = 1.0

#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
#sub = cpp.mesh.MeshFunctionSizet(mesh, mvc)


# Define function spaces
P_z = FiniteElement('P', triangle, 1)
P_omega = VectorElement('P', triangle, 2)
element = MixedElement([P_z, P_omega])
Q_z_omega = FunctionSpace(mesh, element)
Q_z = Q_z_omega.sub(0).collapse()
Q_omega = Q_z_omega.sub(1).collapse()



# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 1.0)'
walls    = 'near(x[1], 0) || near(x[1], 1.0)'

kappa = 1.0
sigma0 = 10.0

z_bottom  = 0.0
z_top = 0.5

omega_bottom = -1.5
omega_top = 1.0
#CHANGE PARAMETERS HERE

#trial analytical expression for the height function z(x,y)
class z_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = z_bottom + (z_top - z_bottom)*x[1]/h
    def value_shape(self):
        return (1,)


class omega_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = omega_bottom + (omega_top - omega_bottom)*x[1]/h
    def value_shape(self):
        return (1,)
    
class sigma_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = sigma0 * x[1]*(h-x[1])/(h**2)
    def value_shape(self):
        return (1,)

#g_{ij}
def g(z):
    return as_tensor([[1, 0],[0, 1+ (z.dx(1))**2]])

def detg(z):
    return ufl.det(g(z))

def sqrt_detg(z):
    return sqrt(detg(z))



bc_z = DirichletBC(Q_z_omega.sub(0), Expression('z_bottom + (z_top - z_bottom)*x[1]/h', degree=1, h = h, z_top = z_top, z_bottom = z_bottom), walls)
bc_omega = DirichletBC(Q_z_omega.sub(1), Expression('omega_bottom + (omega_top - omega_bottom)*x[1]/h', degree=1, h = h, omega_top = omega_top, omega_bottom = omega_bottom), walls)

bc_z_omega = [bc_z, bc_omega]

# Define trial and test functions
nu_z, nu_omega = TestFunctions(Q_z_omega)

# Define functions for solutions at previous and current time steps
z_omega = Function(Q_z_omega)
z, omega = split(z_omega)
sigma = Function(Q_omega)

# Define expressions used in variational forms
n  = FacetNormal(mesh)
kappa = Constant(kappa)
sigma = interpolate(sigma_Expression(element=Q_omega.ufl_element()), Q_omega)



# Define variational problem for step 1
F_z = ( -1.0/(2.0*sqrt_detg(z))* atan(z.dx(1)) * (nu_z.dx(1)) - omega * nu_z ) * sqrt_detg(z) * dx
F_omega = ( kappa * ( (1.0/detg(z)) * (omega.dx(1))* (nu_omega.dx(1)) - 2 * (omega**3) * nu_omega ) + sigma * omega * nu_omega ) *  sqrt_detg(z) * dx
F = F_z + F_omega


# Create XDMF files for visualization output
xdmffile_z = XDMFFile((args.output_directory) + '/z.xdmf')
xdmffile_omega = XDMFFile((args.output_directory) + '/omega.xdmf')

z = interpolate(z_Expression(element=Q_z.ufl_element()), Q_z)
omega = interpolate(omega_Expression(element=Q_omega.ufl_element()), Q_omega)



solve(F == 0, z_omega, bc_z_omega)
    
z_, omega_ = z_omega.split(deepcopy=True)

xdmffile_z.write(z_, 0)
xdmffile_omega.write(omega_, 0)
