'''
run with 

python3 example.py /home/fenics/shared/mesh/membrane_mesh /home/fenics/shared/navier_stokes/fenics_example/solution/

'''


from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import argparse
import ufl as ufl
from geometry import *



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
kappa = 1.0
sigma0 = 1.0


#read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`
mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)

ds_in = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)
ds_out = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=3)
ds_top = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4)
ds_bottom = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=5)

#f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result 
f_test_ds = Function(Q_z)
f_test_ds = interpolate(ScalarFunctionExpression(element=Q_z.ufl_element()), Q_z)

#here I integrate \int ds 1 over the circle and store the result of the integral as a double in inner_circumference
inflow_integral = assemble(f_test_ds*ds_in)
outflow_integral = assemble(f_test_ds*ds_out)
top_wall_integral = assemble(f_test_ds*ds_top)
bottom_wall_integral = assemble(f_test_ds*ds_bottom)
print("Inflow integral = ", inflow_integral, " exact value = 0.807055")
print("Outflow integral = ", outflow_integral, " exact value = 0.227646")
print("Top wall integral = ", top_wall_integral, " exact value = 0.373564")
print("Bottom integral = ", bottom_wall_integral, " exact value = 0.65747")



bc_z = DirichletBC(Q_z_omega.sub(0), Expression('0', degree=0, L=L, h = h), boundary)
bc_omega_in_out = DirichletBC(Q_z_omega.sub(1).sub(0), Expression('0.0', degree=0), in_out_flow)
bc_omega_top_bottom = DirichletBC(Q_z_omega.sub(1).sub(1), Expression('0.0', degree=0), top_bottom_wall)

bc_z_omega = [bc_z, bc_omega_in_out, bc_omega_top_bottom]

# Define trial and test functions
nu_z, nu_omega = TestFunctions(Q_z_omega)

# Define functions for solutions at previous and current time steps
z_omega = Function(Q_z_omega)
z, omega = split(z_omega)
sigma = Function(Q_z)

# Define expressions used in variational forms
n  = FacetNormal(mesh)
kappa = Constant(kappa)
sigma = interpolate(sigma_Expression(element=Q_z.ufl_element()), Q_z)



# Define variational problem for step 1
F_z = ( kappa * ( g_c(omega)[i, j] * (H(omega).dx(j)) * (nu_z.dx(i)) - 2.0 * H(omega) * ( (H(omega))**2 - K(omega) ) * nu_z ) + sigma * H(omega) * nu_z ) * sqrt_detg(omega) * dx
F_omega = ( - z * Nabla_v(nu_omega, omega)[i, i] ) *  sqrt_detg(omega) * dx + \
          ( n_in_out(omega)[i] * g(omega)[i, 1] * z * nu_omega[1] ) * sqrt_deth(omega) * (ds_in + ds_out) + \
          ( n_top_bottom(omega)[i] * g(omega)[i, 0] * z * nu_omega[0] ) * sqrt_deth(omega) * (ds_top + ds_bottom)
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
