from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import argparse
import ufl as ufl



i, j, k, l = ufl.indices(4)


parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
args = parser.parse_args()

set_log_level(30)

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
    
# Create XDMF files for visualization output
xdmffile_u = XDMFFile((args.output_directory) + "/u.xdmf")
xdmffile_v = XDMFFile((args.output_directory) + "/v.xdmf")


# Define function spaces
P_U = FiniteElement('P', triangle, 4)
P_V = FiniteElement('P', triangle, 4)
element = MixedElement([P_U, P_V])
UV = FunctionSpace(mesh, element)
U = UV.sub(0).collapse()
V = UV.sub(1).collapse()




# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 1.0)'
walls    = 'near(x[1], 0) || near(x[1], 1.0)'
# cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
#CHANGE PARAMETERS HERE

# Define inflow profile
g = ('(x[0]*x[0]*x[0]*x[0] + x[1]*x[1]*x[1]*x[1])/48.0')


bcu_inflow = DirichletBC(UV.sub(0), Expression(g, degree=4), inflow)
bcu_outflow = DirichletBC(UV.sub(0), Expression(g, degree=4), outflow)
bcu_walls = DirichletBC(UV.sub(0), Expression(g, degree=4), walls)
# bcu_cylinder = DirichletBC(UV.sub(0), Constant((0, 0)), cylinder)


bc_u = [bcu_inflow, bcu_walls, bcu_outflow]

# Define trial and test functions
nu_u, nu_v = TestFunctions(UV)


# Define functions for solutions at previous and current time steps
uv = TrialFunction(UV)
u, v = split(uv)
uv_ = Function(UV)
u_ = Function(U)
v_ = Function(V)

f = Constant(1.0)



# Define variational problem for step 1
# Fu = rho*dot((u - u_n) / k, nu_u)*dx \
#    + rho*dot(dot(u_n, nabla_grad(u_n)), nu_u)*dx \
#    + inner(sigma(U, p_n), epsilon(nu_u))*dx \
#    + dot(p_n*n, nu_u)*ds - dot(mu*nabla_grad(U)*n, nu_u)*ds \
#    - dot(f, nu_u)*dx
Fu = ( (u.dx(i))*(nu_u.dx(i)) + v*nu_u ) * dx
Fv = ( (v.dx(i)) * (nu_v.dx(i)) + f*nu_v) * dx + (- h*nu_v) * ds
Fuv = Fu + Fv

a = lhs(Fuv)
L = rhs(Fuv)



# Create XDMF files for visualization output
xdmffile_u = XDMFFile('u.xdmf')
xdmffile_v = XDMFFile('v.xdmf')


# Step 1+2
A = assemble(a)
b = assemble(L)
[bc.apply(A) for bc in bc_u]
[bc.apply(b) for bc in bc_u]

solve(A, uv_.vector(), b, 'bicgstab', 'hypre_amg')
    
u_, v_ = uv_.split(deepcopy=True)
   
# Save solution to file (XDMF/HDF5)
xdmffile_u.write(u_, 0)
xdmffile_v.write(v_, 0)