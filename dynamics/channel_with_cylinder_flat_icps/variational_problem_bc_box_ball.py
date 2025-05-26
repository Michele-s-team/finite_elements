from fenics import *
import importlib
import numpy as np
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

T = 1e-3 # final time
num_steps = 10  # number of time steps
dt = T / num_steps  # time step size
mu = 1e-4  # dynamic viscosity
rho = 1  # density

f = Constant((0, 0, 0))

print("L = ", rmsh.L)
print("mu = ", mu)
print("T = ", T)
print("N = ", num_steps)

# Define inflow profile
u_bar_le_profile = Expression((f'4.0*1.5*x[1]*({rmsh.L[1]} - x[1]) / pow({rmsh.L[1]}, 2) * 4.0*1.5*x[2]*({rmsh.L[2]} - x[2]) / pow({rmsh.L[2]}, 2)', '0', '0'), degree=2)

# Define boundary conditions
bc_u_bar_le = DirichletBC(fsp.V, u_bar_le_profile, rmsh.boundary_le)
bc_u_bar_tobo = DirichletBC(fsp.V, Constant((0, 0, 0)), rmsh.boundary_tobo)
bc_u_bar_frba = DirichletBC(fsp.V, Constant((0, 0, 0)), rmsh.boundary_frba)
bc_u_bar_sphere = DirichletBC(fsp.V, Constant((0, 0, 0)), rmsh.boundary_sphere)
bc_p_ri = DirichletBC(fsp.Q, Constant(0), rmsh.boundary_ri)
bc_u_bar = [bc_u_bar_le, bc_u_bar_tobo, bc_u_bar_frba, bc_u_bar_sphere]
bc_p = [bc_p_ri]


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
L2 = dot(nabla_grad(fsp.p_n), nabla_grad(fsp.q)) * rmsh.dx - (1 / dt) * div(fsp.u_bar) * fsp.q * rmsh.dx

# Define variational problem for step 3
a3 = dot(fsp.u, fsp.v) * rmsh.dx
L3 = dot(fsp.u_bar, fsp.v) * rmsh.dx - dt * dot(nabla_grad(fsp.p_ - fsp.p_n), fsp.v) * rmsh.dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bc_u_bar]
[bc.apply(A2) for bc in bc_p]
