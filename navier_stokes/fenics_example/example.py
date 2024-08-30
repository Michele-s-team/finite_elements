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

set_log_level(20)


# # Create mesh
# channel = Rectangle(Point(0, 0), Point(1.0, 1.0))
# cylinder = Circle(Point(0.2, 0.2), 0.05)
# domain = channel - cylinder
# mesh = generate_mesh(domain, 64)

# Create XDMF files for visualization output
xdmffile_z = XDMFFile((args.output_directory) + '/z.xdmf')
xdmffile_omega = XDMFFile((args.output_directory) + '/omega.xdmf')
xdmffile_n = XDMFFile((args.output_directory) + '/n.xdmf')


#read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`
mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)


ds_l = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)
ds_r = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=3)
ds_t = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4)
ds_b = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=5)
ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=6)

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result 
f_test_ds = Function(Q_z)
f_test_ds = interpolate(ScalarFunctionExpression(element=Q_z.ufl_element()), Q_z)

#here I integrate \int ds 1 over the circle and store the result of the integral as a double in inner_circumference
integral_i = assemble(f_test_ds*ds_l)
integral_o = assemble(f_test_ds*ds_r)
integral_t = assemble(f_test_ds*ds_t)
integral_b = assemble(f_test_ds*ds_b)
integral_r = assemble(f_test_ds*ds_circle)

print("Integral i = ", integral_i, " exact value = 1.7302067729935349")
print("Integral o = ", integral_o, " exact value = 1.8395435007455374")
print("Integral t = ", integral_t, " exact value = 1.8015367030205052")
print("Integral b = ", integral_b, " exact value = 1.3427663722292098")
print("Integral R = ", integral_r, " exact value = 2.561571268514012")


n = FacetNormal(mesh)


# Define trial and test functions
nu_z, nu_omega = TestFunctions(Q_z_omega)

# Define functions for solutions at previous and current time steps
z_omega = Function(Q_z_omega)
z, omega = split(z_omega)
sigma = Function(Q_z)
z_0 = Function(Q_z)
omega_0 = Function(Q_omega)


# Define expressions used in variational forms
kappa = Constant(kappa)
sigma = interpolate(sigma_Expression(element=Q_z.ufl_element()), Q_z)
grad_r = interpolate(grad_r_Expression(element=Q_omega.ufl_element()), Q_omega)
grad_R = interpolate(grad_R_Expression(element=Q_omega.ufl_element()), Q_omega)



# Define variational problem for step 1
F_z = ( kappa * ( g_c(omega)[i, j] * (H(omega).dx(j)) * (nu_z.dx(i)) - 2.0 * H(omega) * ( (H(omega))**2 - K(omega) ) * nu_z ) + sigma * H(omega) * nu_z ) * sqrt_detg(omega) * dx \
    - ( \
        ( kappa * (n_lr(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth(omega) * (ds_l + ds_r) + \
        ( kappa * (n_tb(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth(omega) * (ds_t + ds_b) \
    ) 
F_omega = ( - z * Nabla_v(nu_omega, omega)[i, i] - omega[i] * nu_omega[i] ) *  sqrt_detg(omega) * dx + \
          ( (n(omega))[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth(omega) * ds
F_N = eta * ( \
    ( ( n_facet[i]*omega[i] - n_facet[i]*grad_r[i] ) * ( n_facet[k]*g(omega)[k, l]*nu_omega[l] ) ) * ds_r )
F = F_z + F_omega + F_N


#set initial profile of z from analytical expression

z_0 = interpolate(z_Expression(element=Q_z.ufl_element()), Q_z)
omega_0 = interpolate(grad_r_Expression(element=Q_omega.ufl_element()), Q_omega)

 

    
assigner = FunctionAssigner(Q_z_omega, [Q_z, Q_omega])
assigner.assign(z_omega, [z_0, omega_0])
    

#CHANGE PARAMETERS HERE
bc_circle = DirichletBC(Q_z_omega.sub(0), Expression('C', element = Q_z_omega.sub(0).ufl_element(), C = C), boundary_circle)
bc_square = DirichletBC(Q_z_omega.sub(0), Expression('2*C', element = Q_z_omega.sub(0).ufl_element(), C = C), boundary_square)
#CHANGE PARAMETERS HERE
bcs = [bc_circle, bc_square]

solve(F == 0, z_omega, bcs)
    
z_, omega_ = z_omega.split(deepcopy=True)
    
xdmffile_z.write(z_, 0)
xdmffile_omega.write(omega_, 0)
xdmffile_n.write(my_n(), 0)
