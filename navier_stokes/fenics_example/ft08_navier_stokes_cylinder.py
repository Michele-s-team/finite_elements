"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
parser.add_argument("T")
parser.add_argument("N")
args = parser.parse_args()

#T = 1.0            # final time
#N = 1024  # number of time steps
T = (float)(args.T)
N = (int)(args.N)
dt = T / N # time step size
mu = 0.001        # dynamic viscosity
rho = 1            # density

print("T = ", T)
print("N = ", N)

# # Create mesh
# channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
# cylinder = Circle(Point(0.2, 0.2), 0.05)
# domain = channel - cylinder
# mesh = generate_mesh(domain, 64)

L = 2.2
h = 0.41
r = 0.05
c_r = [0.2, 0.2]

#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
#sub = cpp.mesh.MeshFunctionSizet(mesh, mvc)


# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
# inflow   = 'on_boundary && (x[0] < 0.0 + 0.001)'
# outflow  = 'on_boundary && (x[0] > 0.0 + 0.001)'
# cylinder = 'on_boundary && ((x[0]-0.0)*(x[0]-0.0) + (x[1]-0.0)*(x[1]-0.0) < (0.2*0.2))'
#CHANGE PARAMETERS HERE

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

# Define boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)

bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]

# Define trial and test functions
v = TestFunction(V)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u = Function(V)
u_n = Function(V)
u_  = Function(V)
p = Function(Q)
p_n = Function(Q)
p_  = Function(Q)

# Define expressions used in variational forms
U  = 0.5*(u_n + u_)
n  = FacetNormal(mesh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u_ - u_n) / k, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx

# Define variational problem for step 2
F2 = dot(nabla_grad(p), nabla_grad(q))*dx - (dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx)

# Define variational problem for step 3
F3 = dot(u, v)*dx - (dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx)

# Assemble matrices
#A1 = assemble(a1)
#A2 = assemble(a2)
#A3 = assemble(a3)

# Apply boundary conditions to matrices
#[bc.apply(A1) for bc in bcu]
#[bc.apply(A2) for bc in bcp]

# Create XDMF files for visualization output
xdmffile_u = XDMFFile('velocity.xdmf')
xdmffile_p = XDMFFile('pressure.xdmf')

# Create time series (for use in reaction_system.py)
timeseries_u = TimeSeries('velocity_series')
timeseries_p = TimeSeries('pressure_series')

# Save mesh to file (for use in reaction_system.py)
File('cylinder.xml.gz') << mesh

# Create progress bar
# progress = Progress('Time-stepping')
# set_log_level(PROGRESS)

# Time-stepping
print("Starting time iteration ...", flush=True)
t = 0
for n in range(N):

    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_n, t)
    xdmffile_p.write(p_n, t)

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
#    b1 = assemble(L1)
#    [bc.apply(b1) for bc in bcu]
    solve(F1==0, u_, bcu)

    # Step 2: Pressure correction step
#    b2 = assemble(L2)
#    [bc.apply(b2) for bc in bcp]
    solve(F2==0, p, bcp)

    # Step 3: Velocity correction step
#    b3 = assemble(L3)
    solve(F3==0, u)

    # Plot solution
    plot(u_, title='Velocity')
    plot(p_, title='Pressure')



    # Save nodal values to file
    timeseries_u.store(u_.vector(), t)
    timeseries_p.store(p_.vector(), t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p)

    # Update progress bar
    # progress.update(t / T)
    # print('u max:', u_.vector().array().max())
    print("\t%.2f %%" % (100.0*(t/T)), flush=True)

print("... done.", flush=True)

# Hold plot
# interactive()
