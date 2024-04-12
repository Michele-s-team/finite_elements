"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""
#run with clear; clear; python3 navier_stokes_membrane.py [input directory] [output directory]
#ron on mac: clear; clear; python3 navier_stokes_membrane.py /home/fenics/shared/mesh/membrane_mesh /home/fenics/shared/navier_stokes/membrane_simulation/solution
#ron on abacus: clear; clear; python3 navier_stokes_membrane.py /mnt/beegfs/home/mcastel1/navier_stokes /mnt/beegfs/home/mcastel1/navier_stokes/results

from __future__ import print_function
from geometry import *


print("Input directory", args.input_directory)
print("Output directory", args.output_directory)

T = 0.1    # final time
# num_steps = 5000  # number of time steps
num_steps = 10
dt = T / num_steps # time step size
#the Reynolds number, Re = \rho U l / \mu, Re_here = R_{notes fenics}
Re = 10.0
kappa = 1.0

# Create XDMF files for visualization output
xdmffile_v = XDMFFile((args.output_directory) + "/v.xdmf")
xdmffile_w = XDMFFile((args.output_directory) + "/w.xdmf")
xdmffile_sigma = XDMFFile((args.output_directory) + "/sigma.xdmf")
xdmffile_z = XDMFFile((args.output_directory) + "/z.xdmf")
xdmffile_geometry = XDMFFile((args.output_directory) + "/geo.xdmf")

# Create time series (for use in reaction_system.py)
timeseries_v = TimeSeries((args.output_directory) + "/v_series")
timeseries_w = TimeSeries((args.output_directory) + "/w_series")
timeseries_sigma = TimeSeries((args.output_directory) + "/sigma_series")
timeseries_z = TimeSeries((args.output_directory) + "/z_series")




# Create mesh
#channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
#cylinder = Ellipse(Point(0.5, 0.2), 0.05, 0.1)
#cylinder2 = Ellipse(Point(1.5, 0.2), 0.05, 0.1)
#domain = channel - cylinder - cylinder2
#mesh = generate_mesh(domain, 64)




# Define velocity profile on the external boundary
# external_boundary_profile = ('1.0', '0.0')
external_boundary_profile_v = ('1.0', '0.0')
external_boundary_profile_w = '0.0'
# outflow_profile = ('1.0', '0.0')

# Define boundary conditions
#boundary conditions for the velocity u
# bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
# bcu_outflow = DirichletBC(V, Expression(outflow_profile, degree=2), inflow)
bcv_external_boundary = DirichletBC(W, Expression(external_boundary_profile_v, degree=0), external_boundary)
bcv_cylinder = DirichletBC(W, Constant((0, 0)), cylinder)

bcw_external_boundary = DirichletBC(Q2, Expression(external_boundary_profile_w, degree=0), external_boundary)
bcw_cylinder = DirichletBC(Q2, Constant((0)), cylinder)

#boundary conditions for the surface_tension p
# bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bc_v = [bcv_external_boundary, bcv_cylinder]
# bcp = [bcp_outflow]
bc_sigma = []



# Define functions for solutions at previous and current time steps
v_n = Function(W)
v_  = Function(W)
w_n = Function(Q2)
w_  = Function(Q2)
sigma_n = Function(Q2)
sigma_  = Function(Q2)
z_n = Function(Q4)
z_  = Function(Q4)
#a function used to make tests (test the differential operators etc)
f_  = Function(Q4)



#the vector  or function is interpolated  and written into a Function() object
#set the initial conditions for all fields
v_n = interpolate(TangentVelocityExpression(element=W.ufl_element()), W)
w_n = interpolate(NormalVelocityExpression(element=Q2.ufl_element()), Q2)
sigma_n = interpolate(SurfaceTensionExpression(element=Q2.ufl_element()), Q2)
z_n = interpolate(ManifoldExpression(element=Q4.ufl_element()), Q4)


# f_ = interpolate(ScalarFunctionExpression(element=Q2.ufl_element()), Q4)
# z_plot = project(z_, Q)
# grad_z_plot = project(grad_z(z_), V)
# my_vector_field_plot = project(my_vector_field(z_), V)
# detg_plot = project(detg(z_), Q)

# xdmffile_geometry.parameters.update(
#     {
#         "functions_share_mesh": True,
#         "rewrite_function_mesh": False
#     })
# xdmffile_geometry.write(project(v_n, W), 0)
# xdmffile_geometry.write(project(w_n, Q2), 0)
# xdmffile_geometry.write(project(sigma_n, Q2), 0)
# xdmffile_geometry.write(project(z_n, Q4), 0)
# xdmffile_geometry.write(project(normal(z_), V3d), 0)
# xdmffile_geometry.write(project(grad_z(z_), V), 0)
# xdmffile_geometry.write(project(my_vector_field(z_), V), 0)
# xdmffile_geometry.write(project(detg(z_), Q2), 0)
# xdmffile_geometry.write(project(H(z_), Q4), 0)
# xdmffile_geometry.write(project(K(z_), Q4), 0)
# xdmffile_geometry.write(project(Nabla_v(u_, z_)[0,0], Q2), 0)
# xdmffile_geometry.write(project(Nabla_omega(u_, z_)[0,1], Q2), 0)
#here I project Nabla_LB(H,z) on Q4 and not on Q2 because Nabla_LB involves fourth-order derivatives
# xdmffile_geometry.write(project(Nabla_LB(H(z_), z_), Q4), 0)
# xdmffile_geometry.write(project(Nabla_LB2(f_, z_), Q4), 0)
# xdmffile_geometry.write(project(w_, Q2), 0)
# xdmffile_geometry.write(project(d_c(v_, w_, z_)[0,1], Q4), 0)


# xdmffile_z.write(z_, t)
###

#example of how to compute the determinant of a matrix
# A = np.array([[1, 2], [2, 3]])   # Identity tensor
# print("determinant = ", np.linalg.det(A), ".")

# Define expressions used in variational forms
V  = 0.5 * (v_n + v)
Deltat  = Constant(dt)
# mu = Constant(mu)
# rho = Constant(rho)


# v_n[j] * ((v_n[i]).dx(j)) * nu[i]


# Define variational problem for step 1
#step 3 for v
F1v = Re * (dot((v - v_n) / Deltat, nu) * dx \
            + (v_n[j] * Nabla_v(v_n, z_n)[i, j] * nu[i]) * dx \
            - 2.0 * v_n[j] * w_n * g_c(z_n)[i,k] * b(z_n)[k,j] * nu[i] * dx \
            + 0.5 * (w_n**2) * g_c(z_n)[i,j] * Nabla_omega(nu, z_n)[i, j] * dx) \
      + g_c(z_n)[i,j] * Nabla_omega(nu, z_n)[i, j] * sigma_n * dx \
      + 2.0 * d_c(V, w_n, z_n)[i, j] * Nabla_omega(nu, z_n)[i, j] * dx \
    # + dot(sigma_n * n, nu) * ds - dot(2 * epsilon(U) * n, nu) * ds
     # + inner(tensor_sigma(U, sigma_n), epsilon(nu)) * dx
a1v = lhs(F1v)
L1v = rhs(F1v)
#step 3 for w
F1w = ( \
                  Re * ( (w - w_n) / Deltat * o - w_n * ( g_c(z_n)[i, j] * (o.dx(i)) * v_n[j] + o * g_c(z_n)[i, j] * Nabla_omega(v_n, z_n)[i, j] )  )  \
    + 2.0 * kappa * ( - g_c(z_n)[i, j] * H(z_n).dx(i) * o.dx(j) ) \
    + 4.0 * kappa * H(z_n) * ((H(z_n)**2) - K(z_n)) - 2.0 * sigma_n * H(z_n) - 2.0 * ( g_c(z_n)[i, k] * g_c(z_n)[j, l] * Nabla_omega(v_n, z_n)[i, j] * b(z_n)[k, l] \
                                                                                        - 2.0 * w_n * (2.0 * ((H(z_n))**2) - K(z_n))  )  \
        ) * dx
a1w = lhs(F1w)
L1w = rhs(F1w)


# Define variational problem for step 2
F2 = ( \
         g_c(z_n)[i, j] * ( sigma_n.dx(i) - sigma.dx(i) ) * q.dx(j) \
          + (Re / Deltat) * (  g_c(z_n)[i, j]*Nabla_omega(v_, z_n)[i, j] - 2.0 * H(z_n) * w_n ) * q
         \
         ) * dx
a2 = lhs(F2)
L2 = rhs(F2)

# a2 = - dot(nabla_grad(sigma), nabla_grad(q))*dx
# L2 = - dot(nabla_grad(sigma_n), nabla_grad(q)) * dx - (Re / Deltat) * div(v_) * q * dx


# Define variational problem for step 3
#step 3 for v
F3v = ( g_c(z_n)[i,j] * (v_[i] - v[i]) * nu[j] - (Deltat / Re) * g_c(z_n)[i, j] * (sigma_n.dx(i) - sigma_.dx(i)) * nu[j] ) * dx
a3v = lhs(F3v)
L3v = rhs(F3v)
#step 3 for w
F3w = ( (w_ - w) * o - (Deltat / Re) * 2.0 * (sigma_n - sigma_) * H(z_n) * o ) * dx
a3w = lhs(F3w)
L3w = rhs(F3w)

# Define variational problem for step 4
F4 = (normal(z_n)[2] * (z - z_n)  - w_n * Deltat)* zeta * dx
a4 = lhs(F4)
L4 = rhs(F4)




# Assemble matrices
A1v = assemble(a1v)
A1w = assemble(a1w)
A2 = assemble(a2)
A3v = assemble(a3v)
A3w = assemble(a3w)
A4 = assemble(a4)

# Apply boundary conditions to matrices
[bc.apply(A1v) for bc in bc_v]
[bc.apply(A1w) for bc in bc_w]
[bc.apply(A2) for bc in bc_sigma]
[bc.apply(A3v) for bc in bc_v]
[bc.apply(A3w) for bc in bc_w]
#MAYBE HERE I SHOULD ADD AN ANALOGOUS COMMAND FOR Z AND IT BOUNDARY CONDITIONS 



# Save mesh to file (for use in reaction_system.py)
File((args.output_directory) + "/membrane.xml.gz") << mesh

# Create progress bar
#progress = Progress('Time-stepping')
#set_log_level(PROGRESS)


print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
for n in range(num_steps):

    # Write the solution to file
    xdmffile_v.write(v_n, t)
    xdmffile_w.write(w_n, t)
    xdmffile_sigma.write(sigma_n, t)
    xdmffile_z.write(z_n, t)

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1v)
    [bc.apply(b1) for bc in bc_v]
    #this line solves for u^* and stores u^* in u_
    solve(A1v, v_.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: surface_tension correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bc_sigma]
    #this step solves for sigma^{n+1} and stores the solution in sigma_
    solve(A2, sigma_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    b3v = assemble(L3v)
    #this step solves for u^{n+1} and stores the solution in u_. In A3, u_ = u^* from `solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')` and p_n = p_{n+1} from `solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
    solve(A3v, v_.vector(), b3v, 'cg', 'sor')

    # Plot solution
#    plot(u_, title='Velocity')
#    plot(p_, title='surface_tension')



    # Save nodal values to file
    timeseries_v.store(v_.vector(), t)
    timeseries_w.store(w_.vector(), t)
    timeseries_sigma.store(sigma_.vector(), t)
    timeseries_z.store(z_.vector(), t)

    # Update previous solution
    v_n.assign(v_)
    sigma_n.assign(sigma_)

    # Update progress bar
#    progress.update(t / T)
    print("\t%.2f %%" % (100.0*(t/T)), flush=True)

# Hold plot
#interactive()

print("... done.", flush=True)
