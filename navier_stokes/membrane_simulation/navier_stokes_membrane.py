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

T = 0.001    # final time
# num_steps = 5000  # number of time steps
num_steps = 10
dt = T / num_steps # time step size
#the Reynolds number, Re = \rho U l / \mu, Re_here = R_{notes fenics}
Re = 150.0

# Create XDMF files for visualization output
xdmffile_u = XDMFFile((args.output_directory) + "/velocity.xdmf")
xdmffile_p = XDMFFile((args.output_directory) + "/pressure.xdmf")
xdmffile_z = XDMFFile((args.output_directory) + "/z.xdmf")
xdmffile_geometry = XDMFFile((args.output_directory) + "/geometry.xdmf")

# Create time series (for use in reaction_system.py)
timeseries_u = TimeSeries((args.output_directory) + "/velocity_series")
timeseries_p = TimeSeries((args.output_directory) + "/pressure_series")
timeseries_z = TimeSeries((args.output_directory) + "/shape_series")




# Create mesh
#channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
#cylinder = Ellipse(Point(0.5, 0.2), 0.05, 0.1)
#cylinder2 = Ellipse(Point(1.5, 0.2), 0.05, 0.1)
#domain = channel - cylinder - cylinder2
#mesh = generate_mesh(domain, 64)




# Define velocity profile on the external boundary
# external_boundary_profile = ('1.0', '0.0')
external_boundary_profile = ('2.0*(1.0-x[1]*x[1])', '0.0')
# outflow_profile = ('1.0', '0.0')

# Define boundary conditions
#boundary conditions for the velocity u
# bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
# bcu_outflow = DirichletBC(V, Expression(outflow_profile, degree=2), inflow)
bcu_external_boundary = DirichletBC(V, Expression(external_boundary_profile, degree=0), external_boundary)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
#boundary conditions for the pressure p
# bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_external_boundary, bcu_cylinder]
# bcp = [bcp_outflow]
bcp = []



# Define functions for solutions at previous and current time steps
v_n = Function(V)
v_  = Function(V)
w_n = Function(Q2)
w_  = Function(Q2)
sigma_n = Function(Q2)
sigma_  = Function(Q2)
z_n = Function(Q4)
z_  = Function(Q4)
#a function used to make tests (test the differential operators etc)
f_  = Function(Q4)



#the vector  or function is interpolated  and written into a Function() object
v_ = interpolate(MyVectorFunctionExpression(element=V.ufl_element()), V)
z_ = interpolate(SurfaceExpression(element=Q2.ufl_element()), Q2)
w_ = interpolate(NormalVelocityExpression(element=Q2.ufl_element()), Q2)
f_ = interpolate(ScalarFunctionExpression(element=Q2.ufl_element()), Q4)
# z_plot = project(z_, Q)
# grad_z_plot = project(grad_z(z_), V)
# my_vector_field_plot = project(my_vector_field(z_), V)
# detg_plot = project(detg(z_), Q)

xdmffile_geometry.parameters.update(
    {
        "functions_share_mesh": True,
        "rewrite_function_mesh": False
    })
xdmffile_geometry.write(project(z_, Q4), 0)
xdmffile_geometry.write(project(normal(z_), V3d), 0)
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
xdmffile_geometry.write(project(d(v_, w_, z_)[0,1], Q4), 0)


xdmffile_z.write(z_, t)
###

#example of how to compute the determinant of a matrix
# A = np.array([[1, 2], [2, 3]])   # Identity tensor
# print("determinant = ", np.linalg.det(A), ".")

# Define expressions used in variational forms
#U = u^{n+1/2}|_{notes fenics}
U  = 0.5*(v_n + v)
Deltat  = Constant(dt)
# mu = Constant(mu)
# rho = Constant(rho)


v_n[j] * ((v_n[i]).dx(j)) * nu[i]


# Define variational problem for step 1
F1 = Re * (dot((v - v_n) / Deltat, nu) * dx \
           + (v_n[j] * Nabla_v(v_n, z_)[i, j] * nu[i]) * dx) \
     + inner(tensor_sigma(U, sigma_n), epsilon(nu)) * dx \
     + dot(sigma_n * n, nu) * ds - dot(2 * epsilon(U) * n, nu) * ds
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(sigma), nabla_grad(q))*dx
L2 = dot(nabla_grad(sigma_n), nabla_grad(q)) * dx - (Re / Deltat) * div(v_) * q * dx

# Define variational problem for step 3
a3 = dot(v, nu) * dx
L3 = dot(v_, nu) * dx - (Deltat / Re) * dot(nabla_grad(sigma_ - sigma_n), nu) * dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]



# Save mesh to file (for use in reaction_system.py)
File((args.output_directory) + "/membrane.xml.gz") << mesh

# Create progress bar
#progress = Progress('Time-stepping')
#set_log_level(PROGRESS)


print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    #this line solves for u^* and stores u^* in u_
    solve(A1, v_.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    #this step solves for sigma^{n+1} and stores the solution in sigma_
    solve(A2, sigma_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    #this step solves for u^{n+1} and stores the solution in u_. In A3, u_ = u^* from `solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')` and p_n = p_{n+1} from `solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
    solve(A3, v_.vector(), b3, 'cg', 'sor')

    # Plot solution
#    plot(u_, title='Velocity')
#    plot(p_, title='Pressure')

    # Save solution to file (XDMF/HDF5)
    #here u_ = u_{n+1}
    xdmffile_u.write(v_, t)
    #here p_ is p_{n+1}
    xdmffile_p.write(sigma_, t)

    # Save nodal values to file
    timeseries_u.store(v_.vector(), t)
    timeseries_p.store(sigma_.vector(), t)

    # Update previous solution
    v_n.assign(v_)
    sigma_n.assign(sigma_)

    # Update progress bar
#    progress.update(t / T)
    print("\t%.2f %%" % (100.0*(t/T)), flush=True)

# Hold plot
#interactive()

print("... done.", flush=True)
