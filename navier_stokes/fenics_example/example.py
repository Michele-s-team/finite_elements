from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
args = parser.parse_args()

set_log_level(30)


rho = 1            # density

# Create mesh
# channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
# cylinder = Circle(Point(0.2, 0.2), 0.05)
# domain = channel - cylinder
# mesh = generate_mesh(domain, 64)

L = 1
h = 1
# r = 0.05
# c_r = [0.2, 0.2]

#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")


# Define function spaces
P_U = VectorElement('P', triangle, 4)
P_V = FiniteElement('P', triangle, 4)
element = MixedElement([P_U, P_V])
UV = FunctionSpace(mesh, element)
U = UV.sub(0).collapse()
V = UV.sub(1).collapse()



# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
#CHANGE PARAMETERS HERE

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')


bcu_inflow = DirichletBC(UV.sub(0), Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(UV.sub(0), Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(UV.sub(0), Constant((0, 0)), cylinder)

bcp_outflow = DirichletBC(UV.sub(1), Constant(0), outflow)

bc_up = [bcu_inflow, bcu_walls, bcu_cylinder, bcp_outflow]

# Define trial and test functions
v, q = TestFunctions(UV)
vs = TestFunction(U)


# Define functions for solutions at previous and current time steps
up = TrialFunction(UV)
u, p = split(up)

u_n = Function(U)
p_n = Function(V)

up_ = Function(UV)
u_, p_ = split(up_)

us = TrialFunction(U)
ps_ = Function(V)


# Define expressions used in variational forms
U  = 0.5*(u_n + u)
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
F1 = rho*dot((u - u_n) / k, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
# Define variational problem for step 2
F2 = (dot(nabla_grad(p), nabla_grad(q)) - (dot(nabla_grad(p_n), nabla_grad(q)) - (1/k)*div(u)*q))*dx
F12 = F1 + F2

a12 = lhs(F12)
L12 = rhs(F12)


# Define variational problem for step 3
F3 = (dot(us, vs) - (dot(u_, vs) - k*dot(nabla_grad(p_ - p_n), vs))) * dx

a3 = lhs(F3)
L3 = rhs(F3)


# Create XDMF files for visualization output
xdmffile_u = XDMFFile('velocity.xdmf')
xdmffile_p = XDMFFile('pressure.xdmf')

# Create VTK files for visualization output
vtkfile_u = File('v.pvd')
vtkfile_p = File('p.pvd')

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
    A12 = assemble(a12)
    b12 = assemble(L12)
    [bc.apply(A12) for bc in bc_up]
    [bc.apply(b12) for bc in bc_up]

    solve(A12, up_.vector(), b12, 'bicgstab', 'hypre_amg')
    

    # Step 3: Velocity correction step
    # ps_.assign(project(p_, Q))
    A3 = assemble(a3)
    b3 = assemble(L3)
    solve(A3, u_n.vector(), b3,  'cg', 'sor')

    # Save nodal values to file
    timeseries_u.store(u_n.vector(), t)
    timeseries_p.store(p_n.vector(), t)

    # Update previous solution
    #u_n has been already updated by  solve(A3, u_n.vector(), b3,  'cg', 'sor')
    #this step writes the numerical data of up_ into u_n, p_n -> I am interested only in writing into p_n with this line
    _u_, p_n = up_.split(deepcopy=True)
   
 

    print("\t%.2f %%" % (100.0*(t/T)), flush=True)

print("... done.", flush=True)