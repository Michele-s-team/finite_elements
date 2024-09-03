"""
things to fix:


"""
# ron on mac:  clear; clear; python3 navier_stokes_membrane.py /home/fenics/shared/mesh/membrane_mesh /home/fenics/shared/navier_stokes/membrane/solution 1E-2 128
# ron on abacus: clear; clear; python3 navier_stokes_membrane.py /mnt/beegfs/home/mcastel1/navier_stokes /mnt/beegfs/home/mcastel1/navier_stokes/results/  1E-2 128

from __future__ import print_function
from geometry import *

print("Input directory", args.input_directory)
print("Output directory", args.output_directory)

# print("\n\nLinear solver methods:")
# list_linear_solver_methods()
# print("\n\nPreconditioners")
# list_krylov_solver_preconditioners()


#here I integrate \int my_function * ds  over the circle and store the result of the integral as a double in inner_circumference, and similarly for rectangle_integral
circle_integral = assemble(my_function()*ds_circle)
rectangle_integral = assemble(1*ds_rectangle)
print("Circle integral = ", circle_integral)
print("Rectangle integral = ", rectangle_integral)


# T = 1E-2  # final time
# num_steps = 128
T = (float)(args.T)
num_steps = (int)(args.N)
 
dt = T / num_steps  # time step size
# the Reynolds number, Re = \rho U l / \mu, Re_here = R_{notes fenics}
#Re = 1.0
rho = 1.0
mu = 0.001
kappa = 1.0

print("c_r = ", c_r)
# print("c_R = ", c_R)
print("L = ", L)
print("h = ", h)
print("r = ", r)
# print("R = ", R)
print("T = ", T)
print("N = ", num_steps)
print("rho = ", rho)
print("mu = ", mu)
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


# Define functions for solutions at previous and current time steps
vw_n = Function(V_vw)
vw_n_1 = Function(V_vw)
vw_n_2 = Function(V_vw)
vw_ = Function(V_vw)
vs_n = Function(O)


v_n, w_n = vw_n.split()
v_n_1, w_n_1 = vw_n_1.split()
v_n_2, w_n_2 = vw_n_2.split()
v_, w_ = vw_.split()

sigma_n = Function(Q)
sigma_n_1 = Function(Q)
sigma_n_2 = Function(Q)

phi_ = Function(Q)

z_n = Function(Q4)
z_n_1 = Function(Q4)
#z_ = Function(Q4)

## a function used to make tests (test the differential operators etc)
#f_ = Function(Q4)

# the vector  or function is interpolated  and written into a Function() object
# set the initial conditions for all fields
v_n_1 = interpolate(TangentVelocityExpression(element=O.ufl_element()), O)
v_n_2 = v_n_1
w_n_1 = interpolate(NormalVelocityExpression(element=Q.ufl_element()), Q)
w_n_2 = w_n_1
sigma_n_1 = interpolate(SurfaceTensionExpression(element=Q.ufl_element()), Q)
sigma_n_2 = sigma_n_1
z_n = interpolate(ManifoldExpression(element=Q4.ufl_element()), Q4)
z_n_1 = z_n


inflow_profile_v = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')
inflow_profile_w = '0.0'

# Define boundary conditions
# boundary conditions for the velocity u
# bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
# bcu_outflow = DirichletBC(V, Expression(outflow_profile, degree=2), inflow)
bcv_inflow = DirichletBC(V_vw.sub(0), Expression(inflow_profile_v, degree=2), inflow)
bcv_walls = DirichletBC(V_vw.sub(0), Constant((0, 0)), walls)
bcv_cylinder = DirichletBC(V_vw.sub(0), Constant((0, 0)), cylinder)

bcw_inflow = DirichletBC(V_vw.sub(1), Expression(inflow_profile_w, degree=0), inflow)
bcw_walls = DirichletBC(V_vw.sub(1), Constant((0)), walls)
bcw_cylinder = DirichletBC(V_vw.sub(1), Constant((0)), cylinder)

# bcsigma_walls = DirichletBC(Q2, Constant(0), walls)
bc_phi_outflow = DirichletBC(Q, Constant(0), outflow)

# boundary conditions for the surface_tension p
# bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bc_v = [bcv_inflow, bcv_walls, bcv_cylinder]
bc_w = [bcw_inflow, bcw_walls, bcw_cylinder]
bc_phi = [bc_phi_outflow]



# f_ = interpolate(ScalarFunctionExpression(element=Q2.ufl_element()), Q4)
# z_plot = project(z_, Q)
# grad_z_plot = project(grad_z(z_), V)
# my_vector_field_plot = project(my_vector_field(z_), V)
# detg_plot = project(detg(z_), Q)

xdmffile_geo.write(project(z_n, Q4), 0)
# xdmffile_geo.write(project(n_inout(z_n), O), 0)
xdmffile_geo.write(project(n(z_n), O), 0)
#xdmffile_geo.write(project(n_new(z_n), O), 0)
# xdmffile_geo.write(project(detg(z_n), Q2), 0)

# Define expressions used in variational forms
V = 0.5 * (v_n_1 + v)
W = 0.5 * (w_n_1 + w)
Deltat = Constant(dt)
rho = Constant(rho)
mu = Constant(mu)
kappa = Constant(kappa)



# v_n[j] * ((v_n[i]).dx(j)) * nu[i]


# Define variational problem for step 1
# step 1 for v and w
F1v = (rho * ( \
             (v[i] - v_n_1[i])/Deltat \
             + (3.0/2.0 * v_n_1[j] - 1.0/2.0 * v_n_2[j]) * Nabla_v(V, z_n_1)[i, j] \
             - 2.0 * v_n_1[j] * w_n_1 * g_c(z_n_1)[i, k] * b(z_n_1)[k, j] \
             - (3.0/2.0 * w_n_1 - 1.0/2.0 * w_n_2) * g_c(z_n_1)[i, j] * (W.dx(j)) \
             ) * nu[i] \
      + (sigma_n_1 + sigma_n_2)/2.0 * g_c(z_n_1)[i, j] * Nabla_f(nu, z_n_1)[i, j] \
      + 2.0 * mu * d_c(V, W, z_n_1)[j, i] * Nabla_f(nu, z_n_1)[j, i] ) * sqrt_detg(z_n_1) * dx
F1w = (\
       rho * ( \
              ( (w - w_n_1) / Deltat + v_n_1[i] * v_n_1[j] * b(z_n_1)[j,i] ) * omega - W * Nabla_v( (3.0/2.0 * v_n_1 - 1.0/2.0 * v_n_2) * omega , z_n_1)[i, i]  \
              ) \
       + ( \
          kappa * ( - 2.0 * Nabla_LB(H(z_n_1), z_n_1) + 4.0 * H(z_n_1) * ( (H(z_n_1)**2) - K(z_n_1) ) ) - 2.0 * (sigma_n_1 + sigma_n_2)/2.0 * H(z_n_1) \
          - 2.0 * mu * ( g_c(z_n_1)[i, k] * Nabla_v(V, z_n_1)[j, k] * b(z_n_1)[i, j] - 2.0 * W * ( 2.0 * ((H(z_n_1))**2) - K(z_n_1) ) ) ) \
       * omega ) * sqrt_detg(z_n_1) * dx
            
F1 = F1v + F1w
a1 = lhs(F1)
L1 = rhs(F1)

# step 2
F2 = ( \
      g_c(z_n)[i, j] * (phi.dx(i)) * (q.dx(j)) \
      + (rho / Deltat) * (Nabla_v(vs_, z_n_1)[i, i] - 2.0 * H(z_n_1) * ws_) * q \
      ) * sqrt_detg(z_n_1) * dx
a2 = lhs(F2)
L2 = rhs(F2)


# Define variational problem for step 3
# step 3 for v
F3v = ( (vs[i] - vs_[i]) * nus[i] + (Deltat / rho) * g_c(z_n_1)[i, j] * (phi_.dx(j)) * nus[i] ) * sqrt_detg(z_n_1) * dx
a3v = lhs(F3v)
L3v = rhs(F3v)



# Assemble matrices
A1 = assemble(a1)
b1 = assemble(L1)

A2 = assemble(a2)
b2 = assemble(L2)

A3v = assemble(a3v)
b3v = assemble(L3v)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bc_v]
[bc.apply(A1) for bc in bc_w]
[bc.apply(b1) for bc in bc_v]
[bc.apply(b1) for bc in bc_w]
[bc.apply(A2) for bc in bc_phi]


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
    A1 = assemble(a1)
    b1 = assemble(L1)
    
    [bc.apply(A1) for bc in bc_v]
    [bc.apply(A1) for bc in bc_w]
    [bc.apply(b1) for bc in bc_v]
    [bc.apply(b1) for bc in bc_w]
    
    # this line solves for v^* and stores v^* in v_
    solve(A1, vw_.vector(), b1, 'bicgstab', 'hypre_amg')
    


    # Step 2: surface_tension correction step
    vs_.assign(project(v_, O))
    ws_.assign(project(w_, Q))
    
    A2 = assemble(a2)
    b2 = assemble(L2)
    
    [bc.apply(A2) for bc in bc_phi]
    [bc.apply(b2) for bc in bc_phi]
    
    # this step solves for sigma^{n+1} and stores the solution in sigma_
    solve(A2, phi_.vector(), b2, 'bicgstab', 'hypre_amg')
    sigma_n.assign(-2*phi_ + sigma_n_2)


    # Step 3: Velocity correction step
    # step 3 for v
    A3v = assemble(a3v)
    b3v = assemble(L3v)

    vs_.assign(project(v_, O))
    solve(A3v, vs_n.vector(), b3v, 'cg', 'sor')

    v_n.assign(project(vs_n, O))
    w_n.assign(project(w_, Q))

    #step 4: update z
    z_n.assign(z_n_1 + project(dzdt(v_n_1, w_n_1, z_n_1) * Deltat, Q4))
    

    # Update previous solution
    v_n_2.assign(v_n_1)
    v_n_1.assign(v_n)

    w_n_2.assign(w_n_1)
    w_n_1.assign(w_n)

    sigma_n_2.assign(sigma_n_1)
    sigma_n_1.assign(sigma_n)
    
    z_n_1.assign(z_n)

    print("\t%.2f %%" % (100.0 * (t / T)), flush=True)

print("... done.", flush=True)
