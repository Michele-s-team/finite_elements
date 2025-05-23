from fenics import *
import importlib
import numpy as np
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

T = 0.001  # final time
num_steps = 1  # number of time steps
dt = T / num_steps  # time step size
mu = 0.001  # dynamic viscosity
rho = 1  # density

f = Constant((0, 0))

print("L = ", rmsh.L)
print("h = ", rmsh.h)
print("mu = ", mu)
print("T = ", T)
print("N = ", num_steps)

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

# Define boundary conditions
bcu_inflow = DirichletBC(fsp.V, Expression(inflow_profile, degree=2), rmsh.boundary_l)
bcu_walls = DirichletBC(fsp.V, Constant((0, 0)), rmsh.boundary_tb)
bcu_cylinder = DirichletBC(fsp.V, Constant((0, 0)), rmsh.boundary_circle)
bcp_outflow = DirichletBC(fsp.Q, Constant(0), rmsh.boundary_r)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]


# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))


# Define stress tensor
def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))


# Define variational problem for step 1
F1 = rho * dot((fsp.u - fsp.u_n) / dt, fsp.v) * rmsh.dx \
     + rho * dot(dot(fsp.u_n, nabla_grad(fsp.u_n)), fsp.v) * rmsh.dx \
     + inner(sigma(fsp.U, fsp.p_n), epsilon(fsp.v)) * rmsh.dx \
     + dot(fsp.p_n * bgeo.facet_normal, fsp.v) * rmsh.ds - dot(mu * nabla_grad(fsp.U) * bgeo.facet_normal, fsp.v) * rmsh.ds \
     - dot(f, fsp.v) * rmsh.dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(fsp.p), nabla_grad(fsp.q)) * rmsh.dx
L2 = dot(nabla_grad(fsp.p_n), nabla_grad(fsp.q)) * rmsh.dx - (1 / dt) * div(fsp.u_) * fsp.q * rmsh.dx

# Define variational problem for step 3
a3 = dot(fsp.u, fsp.v) * rmsh.dx
L3 = dot(fsp.u_, fsp.v) * rmsh.dx - dt * dot(nabla_grad(fsp.p_ - fsp.p_n), fsp.v) * rmsh.dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]
