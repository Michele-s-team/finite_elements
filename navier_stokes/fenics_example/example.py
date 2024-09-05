'''
this file solves for the steady state of a two-dimensional fluid in the presence of tangential flows

run with
$python3 example.py /home/fenics/shared/mesh/membrane_mesh /home/fenics/shared/navier_stokes/fenics_example/solution/

Note that all sections of the code which need to be changed when an external parameter (e.g., the inflow velocity, the length of the Rectangle, etc...) is changed are bracketed by
#CHANGE PARAMETERS HERE
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

#test for surface elements
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

#print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
print("Integral l = ", integral_l, " exact value = 0.37316849042689265")
print("Integral r = ", integral_r, " exact value = 0.0022778275141919855")
print("Integral t = ", integral_t, " exact value = 1.3656168541307598")
print("Integral b = ", integral_b, " exact value = 1.0283705026372492")
print("Integral circle = ", integral_circle, " exact value = 0.298174235901449")

# Define trial and test functions
nu_sigma, nu_v, nu_z, nu_omega = TestFunctions(Q)


# Define functions for solutions at previous and current time steps
#the function in the total mixed space encorporating the surface tension (sigma), the tangent velocity (v), the membrane height function (z) and the gradient of z (omega)
sigma_v_z_omega = Function(Q)
#the Jacobian
J_sigma_v_z_omega = TrialFunction(Q)
#sigma, v, z, omega are used to store the numerical values of the fields
sigma, v, z, omega = split(sigma_v_z_omega)
#sigma_0, ...., omega_0 are used to store the initial conditions
sigma_0 = Function(Q_sigma)
v_0 = Function(Q_v)
z_0 = Function(Q_z)
omega_0 = Function(Q_omega)


# Define expressions used in variational forms
kappa = Constant(kappa)
rho = Constant(rho)
#the values of \partial_i z = omega_i on the circle and on the square, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
grad_circle = interpolate(grad_circle_Expression(element=Q_omega.ufl_element()), Q_omega)
grad_square = interpolate(grad_square_Expression(element=Q_omega.ufl_element()), Q_omega)



'''
Define variational problem : F_sigma, F_v, F_z and F_omega are related to the PDEs for sigma, ..., omega respecitvely . F_N enforces the BCs with Nitche's method. 
To be safe, I explicitly wrote the each term on each part of the boundary with its own normal vector: for example, on the left (l) and on the right (r) sides of the rectangle, t
he surface elements are ds_l + ds_r, and the normal is n_lr(omega) ~ {+-1 , 0}: this avoids odd interpolations at the corners of the rectangle edges. 
'''

F_sigma = ( (Nabla_v(v, omega)[i, i]) * nu_sigma ) * sqrt_detg(omega) * dx

F_v = ( rho * (  v[j]*Nabla_v(v, omega)[i, j] * nu_v[i] ) + \
       sigma * Nabla_f(nu_v, omega)[i, j]*g_c(omega)[i, j] + \
       2 * eta * d_c(v, omega)[j, i] * Nabla_f(nu_v, omega)[j, i] ) * sqrt_detg(omega) * dx - \
    ( \
        ( sigma * n_lr(omega)[i] * nu_v[i] ) * sqrt_deth_square(omega) * (ds_l + ds_r) + \
        ( sigma * n_tb(omega)[i] * nu_v[i] ) * sqrt_deth_square(omega) * (ds_t + ds_b) + \
        ( sigma * n(omega)[i] * nu_v[i] ) * sqrt_deth_circle(omega, c_r) * ds_circle
    )  - \
    ( \
        ( 2 * eta * d(v, omega)[i, j] * n_lr(omega)[i] * g_c(omega)[j, k] * nu_v[k] ) * sqrt_deth_square(omega) * ds_l + \
        ( 2 * eta * d(v, omega)[i, j] * n_tb(omega)[i] * g_c(omega)[j, k] * nu_v[k] ) * sqrt_deth_square(omega) * (ds_t + ds_b) + \
        ( 2 * eta * d(v, omega)[i, j] * n(omega)[i] * g_c(omega)[j, k] * nu_v[k] ) * sqrt_deth_circle(omega, c_r) * ds_circle
    )

F_z = ( \
        rho * v[i] * v[k] * b(omega)[k, i] * nu_z - \
        kappa * ( 2 * g_c(omega)[i, j] * (H(omega).dx(j)) * (nu_z.dx(i)) - 4.0 * H(omega) * ( (H(omega))**2 - K(omega) ) * nu_z ) - \
            2 * ( sigma * H(omega) + eta * Nabla_v(v, omega)[j, i] * g_c(omega)[i, k] * b(omega)[k, j] ) * nu_z \
        ) * sqrt_detg(omega) * dx \
    + ( \
        ( 2 * kappa * (n_lr(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth_square(omega) * (ds_l + ds_r) + \
        ( 2 * kappa * (n_tb(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth_square(omega) * (ds_t + ds_b) + \
        ( 2 * kappa * (n(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth_circle(omega, c_r) * ds_circle 
    ) 

F_omega = ( z * Nabla_v(nu_omega, omega)[i, i] + omega[i] * nu_omega[i] ) * sqrt_detg(omega) * dx - \
          ( \
            ( (n_lr(omega))[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth_square(omega) * (ds_l + ds_r) + \
            ( (n_tb(omega))[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth_square(omega) * (ds_t + ds_b) + \
            ( (n(omega))[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth_circle(omega, c_r) * ds_circle \
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

#CHANGE PARAMETERS HERE
l_profile_v = Expression(('2.0  *  4.0*1.5*x[1]*(h - x[1]) / pow(h, 2)', '0'), degree=2, h=h)
#CHANGE PARAMETERS HERE

# Define BCs
#BC for sigma on the r edge of the rectangle
bc_sigma_r = DirichletBC(Q.sub(0), Constant(0), boundary_r)

#BCs for v
bc_v_l = DirichletBC(Q.sub(1), l_profile_v, boundary_l)
bc_v_tb = DirichletBC(Q.sub(1), Constant((0, 0)), boundary_tb)
bc_v_circle = DirichletBC(Q.sub(1), Constant((0, 0)), boundary_circle)

#CHANGE PARAMETERS HERE
#BCs for z
bc_z_circle = DirichletBC(Q.sub(2), Expression('0.0', element = Q.sub(2).ufl_element()), boundary_circle)
bc_z_square = DirichletBC(Q.sub(2), Expression('h/4.0', element = Q.sub(2).ufl_element(), h = h), boundary_square)
#CHANGE PARAMETERS HERE

#all boundary conditions collected
bcs = [bc_sigma_r, bc_v_l, bc_v_tb, bc_v_circle, bc_z_circle, bc_z_square]

#solve the variational problem
J  = derivative(F, sigma_v_z_omega, J_sigma_v_z_omega)  # Gateaux derivative in dir. of du
problem = NonlinearVariationalProblem(F, sigma_v_z_omega, bcs, J)
solver  = NonlinearVariationalSolver(problem)
solver.solve()
# solve(F == 0, sigma_v_z_omega, bcs, J)


#get the solution and write it to file
sigma_, v_, z_, omega_ = sigma_v_z_omega.split(deepcopy=True)
    
xdmffile_sigma.write(sigma_, 0)
xdmffile_v.write(v_, 0)
xdmffile_z.write(z_, 0)
xdmffile_omega.write(omega_, 0)

xdmffile_n.write(n_facet_smooth(), 0)