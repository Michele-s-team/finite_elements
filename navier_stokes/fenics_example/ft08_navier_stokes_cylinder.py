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

set_log_level(30)


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

P_V = VectorElement('P', triangle, 2)
P_Q = FiniteElement('P', triangle, 1)
element = MixedElement([P_V, P_Q])
VQ = FunctionSpace(mesh, element)


# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
#CHANGE PARAMETERS HERE

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

# Define boundary conditions
#bc = DirichletBC(V.sub(1), u_D, boundary)


bcu_inflow = DirichletBC(VQ.sub(0), Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(VQ.sub(0), Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(VQ.sub(0), Constant((0, 0)), cylinder)

bcp_outflow = DirichletBC(VQ.sub(1), Constant(0), outflow)

bc_up = [bcu_inflow, bcu_walls, bcu_cylinder, bcp_outflow]

# Define trial and test functions
v, q = TestFunctions(VQ)
vs = TestFunction(V)


# Define functions for solutions at previous and current time steps
up = Function(VQ)
u, p = split(up)


u_n = Function(V)
p_n = Function(Q)
#up_n = Function(VQ)
#u_n, p_n = split(up_n)

up_ = Function(VQ)
u_, p_ = split(up_)

us = Function(V)
ps = Function(Q)


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
F2 = (dot(nabla_grad(p_), nabla_grad(q)) - (dot(nabla_grad(p_n), nabla_grad(q)) - (1/k)*div(u_)*q))*dx
F12 = F1 + F2

# Define variational problem for step 3
F3 = (dot(us, vs) - (dot(u_, vs) - k*dot(nabla_grad(p_ - p_n), vs))) * dx


# Create XDMF files for visualization output
xdmffile_u = XDMFFile('velocity.xdmf')
xdmffile_p = XDMFFile('pressure.xdmf')

# Create time series (for use in reaction_system.py)
timeseries_u = TimeSeries('velocity_series')
timeseries_p = TimeSeries('pressure_series')

# Save mesh to file (for use in reaction_system.py)
File('cylinder.xml.gz') << mesh


# Time-stepping
print("Starting time iteration ...", flush=True)
t = 0
for n in range(N):

    # Save solution to file (XDMF/HDF5)

    xdmffile_u.write(u_n, t)
    xdmffile_p.write(p_n, t)

    # Update current time
    t += dt

    # Step 1+2
    solve(F12 == 0, up_, bc_up)
    
    # _u_, _p_ = up_.split()
    # xdmffile_u.write(_u_, t)
    # xdmffile_p.write(_p_, t)    
    # print(_u_.vector())


    # Step 3: Velocity correction step
#    p_single.assign(project(p_, Q))
    solve(F3 == 0, us)

    # Save nodal values to file
    timeseries_u.store(u_n.vector(), t)
    timeseries_p.store(p_n.vector(), t)

    # Update previous solution
    u_n, p_n = up_.split()
    u_n.assign(us)

    # Update progress bar
    # progress.update(t / T)
    # print('u max:', u_.vector().array().max())
    print("\t%.2f %%" % (100.0*(t/T)), flush=True)

print("... done.", flush=True)

# Hold plot
# interactive()
