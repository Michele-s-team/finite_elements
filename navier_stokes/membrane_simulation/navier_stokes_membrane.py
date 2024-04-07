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
from geometry import *


T = 1    # final time
# num_steps = 5000  # number of time steps
num_steps = 10
dt = T / num_steps # time step size
#the Reynolds number, Re = \rho U l / \mu, Re_here = R_{notes fenics}
Re = 50.0

#paths for mac
input_directory = "/home/fenics/shared/mesh/membrane_mesh"
output_directory = "/home/fenics/shared/navier_stokes/membrane_simulation/solution"

#paths for abacus
# input_directory = "/mnt/beegfs/home/mcastel1/navier_stokes"
# output_directory = "/mnt/beegfs/home/mcastel1/navier_stokes/results"

# Create XDMF files for visualization output
xdmffile_u = XDMFFile(output_directory + "/velocity.xdmf")
xdmffile_p = XDMFFile(output_directory + "/pressure.xdmf")
xdmffile_z = XDMFFile(output_directory + "/z.xdmf")

xdmffile_test = XDMFFile(output_directory + "/test.xdmf")


# Create time series (for use in reaction_system.py)
timeseries_u = TimeSeries(output_directory + "/velocity_series")
timeseries_p = TimeSeries(output_directory + "/pressure_series")
timeseries_z = TimeSeries(output_directory + "/shape_series")


#create mesh with new method
mesh=Mesh()
with XDMFFile(input_directory + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(input_directory + "/line_mesh.xdmf") as infile:
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
#a semi-circle given by the left half of circle_R
inflow   = 'on_boundary && (x[0] < 0.01) && (x[0]*x[0] + x[1]*x[1] > (0.5*0.5))'
#a semi-circle given by the right half of circle_R
outflow   =  'on_boundary && (x[0] > 0.01) && (x[0]*x[0] + x[1]*x[1] > (0.5*0.5))'
#the whole circle_R
external_boundary    = 'on_boundary && (x[0]*x[0] + x[1]*x[1] > (0.5*0.5))'
#the obstacle
cylinder = 'on_boundary && (x[0]*x[0] + x[1]*x[1] < (0.5*0.5))'

# Define velocity profile on the external boundary
external_boundary_profile = ('1.0', '0.0')
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

# Define trial and test functions
#u[i] = v^i_{notes} (tangential velocity)
u = TrialFunction(V)
v = TestFunction(V)
#w = w_notes (normal velocity)
w = TrialFunction(Q)
o = TestFunction(Q)
#p = \sigma_{notes}
p = TrialFunction(Q)
q = TestFunction(Q)
#z = z_notes
z = TrialFunction(Q)
x = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
w_n = Function(Q)
w_  = Function(Q)
p_n = Function(Q)
p_  = Function(Q)
z_n = Function(Q)
z_  = Function(Q)



###
#analytical expression for a vector
class MyVectorFunctionExpression(UserExpression):
    def eval(self, values, x):
        values[0] = x[0]
        values[1] = -x[1]
    def value_shape(self):
        return (2,)
#analytical expression for a function
class MyScalarFunctionExpression(UserExpression):
    def eval(self, values, x):
        values[0] = sin(8*(norm(np.subtract(x, c_r)) - r))*sin(8*(norm(np.subtract(x, c_R)) - R))
    def value_shape(self):
        return (1,)
t=0
#the vector  or function is interpolated  and written into a Function() object
# u_ = interpolate(MyVectorFunctionExpression(element=V.ufl_element()) ,V)
z_ = interpolate(MyScalarFunctionExpression(element=Q.ufl_element()), Q)
xdmffile_test.write(z_, t)
###


# Define expressions used in variational forms
#U = u^{n+1/2}|_{notes fenics}
U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
Deltat  = Constant(dt)
# mu = Constant(mu)
# rho = Constant(rho)

i, j = ufl.indices(2)
Aij = u[i].dx(j)
A = as_tensor(Aij, (i,j))

# Define symmetric gradient
def epsilon(u):
    # nabla_grad(u)_{i,j} = (u[j]).dx[i]
    #sym(nabla_grad(u)) =  nabla_grad(u)_{i,j} + nabla_grad(u)_{j,i}
    # return sym(nabla_grad(u))
    return as_tensor(0.5*(u[i].dx(j) + u[j].dx(i)), (i,j))

# Define stress tensor
def sigma(u, p):
    return as_tensor(2*epsilon(u)[i,j] - p*Identity(len(u))[i,j], (i, j))

# Define variational problem for step 1
#  changed this line to correct error
F1 = dot((u - u_n) / Deltat, v)*dx \
   + Re*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(2*epsilon(U)*n, v)*ds
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/Deltat)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - Deltat*dot(nabla_grad(p_ - p_n), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]



# Save mesh to file (for use in reaction_system.py)
File(output_directory + "/membrane.xml.gz") << mesh

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
    #this line solves for u^* and stores u^* in u_
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    #this step solves for p^{n+1} and stores the solution in p_
    solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    #this step solves for u^{n+1} and stores the solution in u_. In A3, u_ = u^* from `solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')` and p_n = p_{n+1} from `solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    # Plot solution
#    plot(u_, title='Velocity')
#    plot(p_, title='Pressure')

    # Save solution to file (XDMF/HDF5)
    #here u_ = u_{n+1}
    xdmffile_u.write(u_, t)
    #here p_ is p_{n+1}
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
