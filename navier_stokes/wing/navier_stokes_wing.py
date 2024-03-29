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
import meshio
import ufl as ufl


T = 10     # final time
num_steps = 10000  # number of time steps
dt = T / num_steps # time step size
mu = 0.01         # dynamic viscosity
rho = 1            # density
L = 10.0
H = 2.0
# R = 1.0
# L = 1.0
# r = 0.25
#

# def norm(x):
#     return(x[0]*x[0] + x[1]*x[1])
#
# print("****")
# print(norm([1,2]))


#create mesh with new method
mesh=Mesh()
with XDMFFile("/home/fenics/shared/mesh/wing/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("/home/fenics/shared/mesh/wing/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
#sub = cpp.mesh.MeshFunctionSizet(mesh, mvc)



# Create mesh
#channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
#cylinder = Ellipse(Point(0.5, 0.2), 0.05, 0.1)
#cylinder2 = Ellipse(Point(1.5, 0.2), 0.05, 0.1)
#domain = channel - cylinder - cylinder2
#mesh = generate_mesh(domain, 64)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow   = 'on_boundary && near(x[0], 0.0)'
outflow   =  'on_boundary && near(x[0], 10.0)'
walls    = 'near(x[1], 0.0) || near(x[1], 2.0)'
cylinder = 'on_boundary && ((x[0]>1.0) && (x[0]<5.0) && (x[1]>0.1) && (x[1]<1.9))'

# Define inflow profile
inflow_profile = ('1.0', '0.0')
# outflow_profile = ('1.0', '0.0')

# Define boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
# bcu_outflow = DirichletBC(V, Expression(outflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

i, j = ufl.indices(2)
Aij = u[i].dx(j)
A = as_tensor(Aij, (i,j))

# Define symmetric gradient
def epsilon(u):
    # nabla_grad(u)_{i,j} = (u[j]).dx[i]
    #sym(nabla_grad(u)) =  nabla_grad(u)_{i,j} + nabla_grad(u)_{j,i}
    # return sym(nabla_grad(u))
    return as_tensor(u[i].dx(j) + u[j].dx(i), (i,j))

# Define stress tensor
def sigma(u, p):
    return as_tensor(2*mu*epsilon(u)[i,j] - p*Identity(len(u))[i,j], (i, j))

# Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Create XDMF files for visualization output
xdmffile_u = XDMFFile('solution/velocity.xdmf')
xdmffile_p = XDMFFile('solution/pressure.xdmf')

# Create time series (for use in reaction_system.py)
timeseries_u = TimeSeries('solution/velocity_series')
timeseries_p = TimeSeries('solution/pressure_series')

# Save mesh to file (for use in reaction_system.py)
File('solution/cylinder.xml.gz') << mesh

# Create progress bar
#progress = Progress('Time-stepping')
#set_log_level(PROGRESS)

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    # Plot solution
#    plot(u_, title='Velocity')
#    plot(p_, title='Pressure')

    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t)

    # Save nodal values to file
    timeseries_u.store(u_.vector(), t)
    timeseries_p.store(p_.vector(), t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

    # Update progress bar
#    progress.update(t / T)
    print("%.2f %%" % (100.0*(t/T)))

# Hold plot
#interactive()
