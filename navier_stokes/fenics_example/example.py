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
xdmffile_sigma = XDMFFile((args.output_directory) + '/sigma.xdmf')
xdmffile_v = XDMFFile((args.output_directory) + '/v.xdmf')
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
integral_l = assemble(f_test_ds*ds_l)
integral_r = assemble(f_test_ds*ds_r)
integral_t = assemble(f_test_ds*ds_t)
integral_b = assemble(f_test_ds*ds_b)
integral_circle = assemble(f_test_ds*ds_circle)

print("Integral l = ", integral_l, " exact value = 1.7302067729935349")
print("Integral r = ", integral_r, " exact value = 1.8395435007455374")
print("Integral t = ", integral_t, " exact value = 1.8015367030205052")
print("Integral b = ", integral_b, " exact value = 1.3427663722292098")
print("Integral circle = ", integral_circle, " exact value = 2.561571268514012")

# Define trial and test functions
nu_sigma, nu_v, nu_z, nu_omega = TestFunctions(Q)


# Define functions for solutions at previous and current time steps
sigma_v_z_omega = Function(Q)
sigma, v, z, omega = split(sigma_v_z_omega)
sigma_0 = Function(Q_sigma)
v_0 = Function(Q_v)
z_0 = Function(Q_z)
omega_0 = Function(Q_omega)


# Define expressions used in variational forms
kappa = Constant(kappa)
rho = Constant(rho)
grad_circle = interpolate(grad_circle_Expression(element=Q_omega.ufl_element()), Q_omega)
grad_square = interpolate(grad_square_Expression(element=Q_omega.ufl_element()), Q_omega)



# Define variational problem 

F_sigma = ( (Nabla_v(v, omega)[i, i]) * nu_sigma ) * sqrt_detg(omega) * dx

F_v = ( rho * (  v[j]*Nabla_v(v, omega)[i, j] * nu_v[i] ) + \
       sigma * Nabla_f(nu_v, omega)[i, j]*g_c(omega)[i, j] + \
       2 * eta * d_c(v, omega)[j, i] * Nabla_f(nu_v, omega)[j, i] ) * sqrt_detg(omega) * dx - \
    ( \
        ( sigma * n_lr(omega)[i] * nu_v[i] ) * sqrt_deth_square(omega) * (ds_l + ds_r) + \
        ( sigma * n_tb(omega)[i] * nu_v[i] ) * sqrt_deth_square(omega) * (ds_t + ds_b) + \
        ( sigma * n(omega)[i] * nu_v[i] ) * sqrt_deth_circle(omega) * ds_circle
    )  - \
    ( \
        ( 2 * eta * d(v, omega)[i, j] * n_lr(omega)[i] * g_c(omega)[j, k] * nu_v[k] ) * sqrt_deth_square(omega) * ds_l + \
        ( 2 * eta * d(v, omega)[i, j] * n_tb(omega)[i] * g_c(omega)[j, k] * nu_v[k] ) * sqrt_deth_square(omega) * (ds_t + ds_b) + \
        ( 2 * eta * d(v, omega)[i, j] * n(omega)[i] * g_c(omega)[j, k] * nu_v[k] ) * sqrt_deth_circle(omega) * ds_circle
    )

F_z = ( \
        rho * v[i] * v[k] * b(omega)[k, i] * nu_z - \
        kappa * ( 2 * g_c(omega)[i, j] * (H(omega).dx(j)) * (nu_z.dx(i)) - 4.0 * H(omega) * ( (H(omega))**2 - K(omega) ) * nu_z ) - \
            2 * ( sigma * H(omega) + eta * Nabla_v(v, omega)[j, i] * g_c(omega)[i, k] * b(omega)[k, j] ) * nu_z \
        ) * sqrt_detg(omega) * dx \
    + ( \
        ( 2 * kappa * (n_lr(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth_square(omega) * (ds_l + ds_r) + \
        ( 2 * kappa * (n_tb(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth_square(omega) * (ds_t + ds_b) + \
        ( 2 * kappa * (n(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth_circle(omega) * ds_circle 
    ) 

F_omega = ( z * Nabla_v(nu_omega, omega)[i, i] + omega[i] * nu_omega[i] ) * sqrt_detg(omega) * dx - \
          ( \
            ( (n_lr(omega))[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth_square(omega) * (ds_l + ds_r) + \
            ( (n_tb(omega))[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth_square(omega) * (ds_t + ds_b) + \
            ( (n(omega))[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth_circle(omega) * ds_circle \
          )

F_N = alpha * (  \
        ( ( (n_facet_lr())[i]*omega[i] - (n_facet_lr())[i]*grad_square[i] ) * ( (n_facet_lr())[k]*g(omega)[k, l]*nu_omega[l] ) ) * (ds_l + ds_r) + \
        ( ( (n_facet_tb())[i]*omega[i] - (n_facet_tb())[i]*grad_square[i] ) * ( (n_facet_tb())[k]*g(omega)[k, l]*nu_omega[l] ) ) * (ds_t + ds_b) + \
        ( ( n_facet[i]*omega[i] - n_facet[i]*grad_circle[i] ) * ( n_facet[k]*g(omega)[k, l]*nu_omega[l] ) ) * ds_circle \
    )

F = ( F_sigma + F_v + F_z + F_omega ) + F_N

#set initial profile of fields 
sigma_0 = interpolate(sigma0_Expression(element=Q_sigma.ufl_element()), Q_sigma)
v_0 = interpolate(v0_Expression(element=Q_v.ufl_element()), Q_v)
z_0 = interpolate(z0_Expression(element=Q_z.ufl_element()), Q_z)
omega_0 = interpolate(omega0_Expression(element=Q_omega.ufl_element()), Q_omega)

    
assigner = FunctionAssigner(Q, [Q_sigma, Q_v, Q_z, Q_omega])
assigner.assign(sigma_v_z_omega, [sigma_0, v_0, z_0, omega_0])

l_profile_v = Expression(('4.0*1.5*x[1]*(h - x[1]) / pow(h, 2)', '0'), degree=2, h=h)

# Define boundary conditions
# boundary conditions for the velocity u
bc_v_l = DirichletBC(O, l_profile_v, boundary_l)
bc_v_tb = DirichletBC(O, Constant((0, 0)), boundary_tb)
bc_v_circle = DirichletBC(O, Constant((0, 0)), boundary_circle)

bc_sigma_r = DirichletBC(Q, Constant(0), boundary_r)

# boundary conditions for the surface_tension p
bc_v = [bc_v_l, bc_v_tb, bc_v_circle]
bc_sigma = [bc_sigma_r]

    
'''

#CHANGE PARAMETERS HERE
bc_circle = DirichletBC(Q.sub(0), Expression('0.5 * C', element = Q.sub(0).ufl_element(), C = C), boundary_circle)
bc_square = DirichletBC(Q.sub(0), Expression('C', element = Q.sub(0).ufl_element(), C = C), boundary_square)
#CHANGE PARAMETERS HERE
bcs = [bc_circle, bc_square]

solve(F == 0, z_omega, bcs)
    
z_, omega_ = z_omega.split(deepcopy=True)
    
xdmffile_z.write(z_, 0)
xdmffile_omega.write(omega_, 0)
xdmffile_n.write(n_facet_smooth(), 0)
'''