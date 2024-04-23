"""
things to fix:


"""
# run with clear; clear; python3 navier_stokes_membrane.py [input directory] [output directory]
# ron on mac: clear; clear; python3 navier_stokes_membrane.py /home/fenics/shared/mesh/membrane_mesh /home/fenics/shared/navier_stokes/membrane_simulation/solution
# ron on abacus: clear; clear; python3 navier_stokes_membrane.py /mnt/beegfs/home/mcastel1/navier_stokes /mnt/beegfs/home/mcastel1/navier_stokes/results

from __future__ import print_function
from geometry import *

print("Input directory", args.input_directory)
print("Output directory", args.output_directory)

# print("\n\nLinear solver methods:")
# list_linear_solver_methods()
# print("\n\nPreconditioners")
# list_krylov_solver_preconditioners()


T = 0.1  # final time
num_steps = 100
dt = T / num_steps  # time step size
# the Reynolds number, Re = \rho U l / \mu, Re_here = R_{notes fenics}
Re = 1.0
kappa = 1.0

print("c_r = ", c_r)
# print("c_R = ", c_R)
print("L = ", L)
print("h = ", h)
print("r = ", r)
# print("R = ", R)
print("T = ", T)
print("Number of steps = ", num_steps)
print("Re = ", Re)
print("kappa = ", kappa)

# Create XDMF files for visualization output
xdmffile_v = XDMFFile((args.output_directory) + "/v.xdmf")
xdmffile_w = XDMFFile((args.output_directory) + "/w.xdmf")
xdmffile_sigma = XDMFFile((args.output_directory) + "/sigma.xdmf")
xdmffile_z = XDMFFile((args.output_directory) + "/z.xdmf")

xdmffile_geo = XDMFFile((args.output_directory) + "/geo.xdmf")
# this is needed to write multiple data series to xdmffile_geo
xdmffile_geo.parameters.update(
    {
        "functions_share_mesh": True,
        "rewrite_function_mesh": False
    })

# Create time series (for use in reaction_system.py)
timeseries_v = TimeSeries((args.output_directory) + "/v_series")
timeseries_w = TimeSeries((args.output_directory) + "/w_series")
timeseries_sigma = TimeSeries((args.output_directory) + "/sigma_series")
timeseries_z = TimeSeries((args.output_directory) + "/z_series")

# Create mesh
# channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
# cylinder = Ellipse(Point(0.5, 0.2), 0.05, 0.1)
# cylinder2 = Ellipse(Point(1.5, 0.2), 0.05, 0.1)
# domain = channel - cylinder - cylinder2
# mesh = generate_mesh(domain, 64)


# Define velocity profile on the external boundary
# external_boundary_profile = ('1.0', '0.0')
inflow_profile_v = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')
inflow_profile_w = '0.0'
# outflow_profile = ('1.0', '0.0')

# Define boundary conditions
# boundary conditions for the velocity u
# bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
# bcu_outflow = DirichletBC(V, Expression(outflow_profile, degree=2), inflow)
bcv_inflow = DirichletBC(O, Expression(inflow_profile_v, degree=2), inflow)
bcv_walls = DirichletBC(O, Constant((0, 0)), walls)
bcv_cylinder = DirichletBC(O, Constant((0, 0)), cylinder)

bcw_inflow = DirichletBC(Q, Expression(inflow_profile_w, degree=0), inflow)
bcw_walls = DirichletBC(Q, Constant((0)), walls)
bcw_cylinder = DirichletBC(Q, Constant((0)), cylinder)

# bcsigma_walls = DirichletBC(Q2, Constant(0), walls)
bcsigma_outflow = DirichletBC(Q, Constant(0), outflow)

# boundary conditions for the surface_tension p
# bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bc_v = [bcv_inflow, bcv_walls, bcv_cylinder]
bc_w = [bcw_inflow, bcw_cylinder, bcw_cylinder]
bc_sigma = [bcsigma_outflow]

# Define functions for solutions at previous and current time steps
v_n = Function(O)
v_ = Function(O)
w_n = Function(Q)
w_ = Function(Q)
sigma_n = Function(Q)
sigma_ = Function(Q)
z_n = Function(Q4)
z_ = Function(Q4)
# a function used to make tests (test the differential operators etc)
f_ = Function(Q4)

# the vector  or function is interpolated  and written into a Function() object
# set the initial conditions for all fields
v_n = interpolate(TangentVelocityExpression(element=O.ufl_element()), O)
w_n = interpolate(NormalVelocityExpression(element=Q.ufl_element()), Q)
sigma_n = interpolate(SurfaceTensionExpression(element=Q.ufl_element()), Q)
z_n = interpolate(ManifoldExpression(element=Q4.ufl_element()), Q4)

# f_ = interpolate(ScalarFunctionExpression(element=Q2.ufl_element()), Q4)
# z_plot = project(z_, Q)
# grad_z_plot = project(grad_z(z_), V)
# my_vector_field_plot = project(my_vector_field(z_), V)
# detg_plot = project(detg(z_), Q)

xdmffile_geo.write(project(z_n, Q4), 0)
xdmffile_geo.write(project(n_inout(z_n), O), 0)
# xdmffile_geo.write(project(n(z_n), O), 0)
# xdmffile_geo.write(project(detg(z_n), Q2), 0)
# xdmffile_geo.write(project(H(z_n), Q4), 0)
# xdmffile_geo.write(project(K(z_n), Q4), 0)
# xdmffile_geo.write(project(grad_z(z_), V), 0)
# xdmffile_geo.write(project(my_vector_field(z_), V), 0)
# xdmffile_geo.write(project(Nabla_v(u_, z_)[0,0], Q2), 0)
# xdmffile_geo.write(project(Nabla_omega(u_, z_)[0,1], Q2), 0)
# here I project Nabla_LB(H,z) on Q4 and not on Q2 because Nabla_LB involves fourth-order derivatives
# xdmffile_geo.write(project(Nabla_LB(H(z_), z_), Q4), 0)
# xdmffile_geo.write(project(Nabla_LB2(f_, z_), Q4), 0)
# xdmffile_geo.write(project(w_, Q2), 0)
# xdmffile_geo.write(project(d_c(v_, w_, z_)[0,1], Q4), 0)


# xdmffile_z.write(z_, t)
###

# example of how to compute the determinant of a matrix
# A = np.array([[1, 2], [2, 3]])   # Identity tensor
# print("determinant = ", np.linalg.det(A), ".")

# Define expressions used in variational forms
V = 0.5 * (v_n + v)
Deltat = Constant(dt)
Re = Constant(Re)
kappa = Constant(kappa)
# mu = Constant(mu)
# rho = Constant(rho)


# v_n[j] * ((v_n[i]).dx(j)) * nu[i]


# Define variational problem for step 1
# step 1 for v
F1v = Re * ( \
            (dot((v - v_n) / Deltat, nu) \
             + (v_n[j] * Nabla_v(v_n, z_n)[i, j] * nu[i]) \
             - 2.0 * v_n[j] * w_n * g_c(z_n)[i, k] * b(z_n)[k, j] * nu[i] \
             + 0.5 * (w_n ** 2) * g_c(z_n)[i, j] * Nabla_f(nu, z_n)[i, j]) * sqrt_detg(z_n) * dx \
            + (- 0.5 * (w_n ** 2) * nu[i] * n_inout(z_n)[i]) * sqrt_deth(z_n) * ds) \
      + (g_c(z_n)[i, j] * Nabla_f(nu, z_n)[i, j] * sigma_n \
         + 2.0 * d_c(V, w_n, z_n)[i, j] * Nabla_f(nu, z_n)[i, j]) * sqrt_detg(z_n) * dx \
      + (- sigma_n * nu[i] * n_inout(z_n)[i] - 2.0 * d_c(V, w_n, z_n)[i, j] * nu[j] * g(z_n)[i, k] * n_inout(z_n)[k]) * sqrt_deth(
    z_n) * ds
# + dot(sigma_n * n, nu) * sqrt_detg(z_n) * ds - dot(2 * epsilon(U) * n, nu) * sqrt_detg(z_n) * ds
# + inner(tensor_sigma(U, sigma_n), epsilon(nu)) * sqrt_detg(z_n) * dx
a1v = lhs(F1v)
L1v = rhs(F1v)
# step 1 for w
F1w = (Re * ((w - w_n) / Deltat * omega - w_n * ((omega.dx(i)) * v_n[i] + omega * Nabla_v(v_n, z_n)[i, i])) * sqrt_detg(
    z_n) * dx + (w_n * omega * v_n[i] * g(z_n)[i, j] * n_inout(z_n)[j]) * sqrt_deth(z_n) * ds) \
      + (2.0 * kappa * (- g_c(z_n)[i, j] * H(z_n).dx(i) * omega.dx(j)) \
         + (4.0 * kappa * H(z_n) * ((H(z_n) ** 2) - K(z_n)) - 2.0 * sigma_n * H(z_n) - 2.0 * (
                g_c(z_n)[k, i] * Nabla_v(v_n, z_n)[j, k] * b(z_n)[i, j] \
                - 2.0 * w_n * (2.0 * ((H(z_n)) ** 2) - K(z_n)))) * omega) * sqrt_detg(z_n) * dx \
      + (2.0 * kappa * omega * (H(z_n).dx(i)) * n_inout(z_n)[i]) * sqrt_deth(z_n) * ds
a1w = lhs(F1w)
L1w = rhs(F1w)

# step 2
F2 = ( \
                 g_c(z_n)[i, j] * ((sigma_n - sigma).dx(i)) * q.dx(j) \
                 + (Re / Deltat) * (Nabla_v(v_, z_n)[i, i] - 2.0 * H(z_n) * w_n) * q \
         ) * sqrt_detg(z_n) * dx
    #THIS TERM IS ZERO BECAUSE WE ASSUME THAT ON THE BOUNDARIES WHERE v IS SPECIFIED, WE HAVE d sigma / d n  = 0
     # - (((sigma_n - sigma).dx(i)) * n(z_n)[i] * q) * sqrt_deth(z_n) * ds
    # THIS TERM IS ZERO BECAUSE WE ASSUME THAT ON THE BOUNDARIES WHERE v IS SPECIFIED, WE HAVE d sigma / d n  = 0
a2 = lhs(F2)
L2 = rhs(F2)

# a2 = - dot(nabla_grad(sigma), nabla_grad(q))*dx
# L2 = - dot(nabla_grad(sigma_n), nabla_grad(q)) * dx - (Re / Deltat) * div(v_) * q * dx


# Define variational problem for step 3
# step 3 for v
F3v = ((v_[i] - v[i]) * nu[i] - (Deltat / Re) * g_c(z_n)[i, j] * ((sigma_n - sigma_).dx(i)) * nu[j]) * sqrt_detg(
    z_n) * dx
a3v = lhs(F3v)
L3v = rhs(F3v)
# step 3 for w
F3w = ((w_ - w) * omega - (Deltat / Re) * 2.0 * (sigma_n - sigma_) * H(z_n) * omega) * sqrt_detg(z_n) * dx
a3w = lhs(F3w)
L3w = rhs(F3w)

# Define variational problem for step 4
F4 = (normal(z_n)[2] * (z - z_n) - w_n * Deltat) * zeta * sqrt_detg(z_n) * dx
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
# [bc.apply(A3v) for bc in bc_v]
# [bc.apply(A3w) for bc in bc_w]
# MAYBE HERE I SHOULD ADD AN ANALOGOUS COMMAND FOR Z AND IT BOUNDARY CONDITIONS


# Save mesh to file (for use in reaction_system.py)
File((args.output_directory) + "/membrane.xml.gz") << mesh

# Create progress bar
# progress = Progress('Time-stepping')
# set_log_level(PROGRESS)


print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
for n in range(num_steps):

    # Write the solution to file
    xdmffile_v.write(v_n, t)
    xdmffile_w.write(w_n, t)
    xdmffile_sigma.write(sigma_n, t)
    xdmffile_z.write(z_n, t)

    # Save nodal values to file
    timeseries_v.store(v_n.vector(), t)
    timeseries_w.store(w_n.vector(), t)
    timeseries_sigma.store(sigma_n.vector(), t)
    timeseries_z.store(z_n.vector(), t)

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    # step 1 for v
    b1v = assemble(L1v)
    [bc.apply(b1v) for bc in bc_v]
    # this line solves for v^* and stores v^* in v_
    solve(A1v, v_.vector(), b1v, 'bicgstab', 'hypre_amg')
    # step 1 for w
    b1w = assemble(L1w)
    [bc.apply(b1w) for bc in bc_w]
    # this line solves for w^* and stores w^* in w_
    solve(A1w, w_.vector(), b1w, 'bicgstab', 'hypre_amg')

    # Step 2: surface_tension correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bc_sigma]
    # this step solves for sigma^{n+1} and stores the solution in sigma_
    solve(A2, sigma_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    # step 3 for v
    b3v = assemble(L3v)
    # this step solves for v^{n+1} and stores the solution in v_. In A3v, v_ = v^* from `solve(A1v, v_.vector(), b1v, 'bicgstab', 'hypre_amg')` and sigma_n = sigma_{n+1} from `solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
    solve(A3v, v_.vector(), b3v, 'cg', 'sor')
    # step 3 for w
    b3w = assemble(L3w)
    # this step solves for w^{n+1} and stores the solution in w_. In A3w, w_ = w^* from `solve(A1w, w_.vector(), b1w, 'bicgstab', 'hypre_amg')` and sigma_n = sigma_{n+1} from `solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
    solve(A3w, w_.vector(), b3w, 'cg', 'sor')

    # step 4
    # Update previous solution
    v_n.assign(v_)
    w_n.assign(w_)
    sigma_n.assign(sigma_)
    z_n.assign(z_n + project(dzdt(v_, w_, z_n) * Deltat, Q4))

    print("\t%.2f %%" % (100.0 * (t / T)), flush=True)

print("... done.", flush=True)
