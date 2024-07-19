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

L = 1.0
h = 1.0
kappa = 1.0

print("N = ", N)

# Create XDMF files for visualization output
xdmffile_z = XDMFFile((args.output_directory) + '/z.xdmf')
xdmffile_omega = XDMFFile((args.output_directory) + '/omega.xdmf')



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



# Define variational problem for step 1
F_z = ( kappa * ( g_c(omega)[i, j] * (H(omega).dx(j)) * (nu_z.dx(i)) - 2.0 * H(omega) * ( (H(omega))**2 - K(omega) ) * nu_z ) + sigma * H(omega) * nu_z ) * sqrt_detg(omega) * dx \
    - ( kappa * (n_in_out(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth(omega) * (ds_in + ds_out) \
    - ( kappa * (n_top_bottom(omega))[i] * nu_z * (H(omega).dx(i)) ) * sqrt_deth(omega) * (ds_top + ds_bottom)
F_omega = ( - z * Nabla_v(nu_omega, omega)[i, i] - omega[i] * nu_omega[i] ) *  sqrt_detg(omega) * dx + \
          ( n_in_out(omega)[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth(omega) * (ds_in + ds_out) + \
          ( n_top_bottom(omega)[i] * g(omega)[i, j] * z * nu_omega[j] ) * sqrt_deth(omega) * (ds_top + ds_bottom)
F = F_z + F_omega



# Read the contents of the file back into a new function, `z_read`:
# z_read = Function(Q_z)
# file_read = HDF5File(MPI.comm_world, "z_saved.h5", "r")
# file_read.read(z_read, "/f")
# file_read.close()
# # Read the contents of the file back into a new function, `omega_read`:
# omega_read = Function(Q_omega)
# file_read = HDF5File(MPI.comm_world, "omega_saved.h5", "r")
# file_read.read(omega_read, "/f")
# file_read.close()

# assigner = FunctionAssigner(Q_z_omega, [Q_z, Q_omega])
# assigner.assign(z_omega, [z_read, omega_read])

#rescale the solution for the next iteration
# z_read.assign(7./6.*z_read)
# omega_read.assign(7./6.*omega_read)

#set initial profile of z from analytical expression

z_0 = interpolate(z_Expression(element=Q_z.ufl_element()), Q_z)
omega_0 = interpolate(omega_Expression(element=Q_omega.ufl_element()), Q_omega)


for n in range(N):
    
    print("Step #", n)    
    
    C = C_n(n)
    
    print("C = ", C)    

    
    assigner = FunctionAssigner(Q_z_omega, [Q_z, Q_omega])
    assigner.assign(z_omega, [z_0, omega_0])

    #CHANGE PARAMETERS HERE
    bc_z = DirichletBC(Q_z_omega.sub(0), Expression('C * cos(2*pi*x[0]/L) * pow(sin(2*pi*(x[1]+x[0])/h), 2)', element = Q_z_omega.sub(0).ufl_element(), C = C, L=L, h=h), boundary)
    bc_omega_in_out = DirichletBC(Q_z_omega.sub(1).sub(0), Expression('C * cos(2*pi*x[0]/L) * pow(x[1], 2)/2.0', element = Q_z_omega.sub(1).sub(0).ufl_element(), C = C, L=L, h=h), in_out_flow)
    bc_omega_top_bottom = DirichletBC(Q_z_omega.sub(1).sub(1), Expression('C * sin(2*pi*x[0]/L) * cos(pi*x[1]/h)', element = Q_z_omega.sub(1).sub(1).ufl_element(), C = C, L=L, h=h), top_bottom_wall)
    #CHANGE PARAMETERS HERE
    bc_z_omega = [bc_z, bc_omega_in_out, bc_omega_top_bottom]

    solve(F == 0, z_omega, bc_z_omega)
    
    z_, omega_ = z_omega.split(deepcopy=True)
    z_0.assign(project(C_n(n+1)/C_n(n)*z_, Q_z))
    omega_0.assign(project(C_n(n+1)/C_n(n)*omega_, Q_omega))
    
    xdmffile_z.write(z_, n)
    xdmffile_omega.write(omega_, n)



# # Write `z_` to a file:
# z_saved = Function(Q_z)
# z_saved = project(z_, Q_z)
# file_write = HDF5File(MPI.comm_world, "z_saved.h5", "w")
# file_write.write(z_saved, "/f")
# file_write.close()
# # Write `omega_` to a file:
# omega_saved = Function(Q_omega)
# omega_saved = project(omega_, Q_omega)
# file_write = HDF5File(MPI.comm_world, "omega_saved.h5", "w")
# file_write.write(omega_saved, "/f")
# file_write.close()



# #write const x z to output.xdmf
# z_write = Function(Q_z)
# z_write = project(3*z_read, Q_z)
# xdmffile_f = XDMFFile('output_z.xdmf')
# xdmffile_f.write(z_write, 0)
# #write const x omega to output.xdmf
# omega_write = Function(Q_omega)
# omega_write = project(3*omega_read, Q_omega)
# xdmffile_f = XDMFFile('output_omega.xdmf')
# xdmffile_f.write(omega_write, 0)

