'''
this module solves for the fields, v^n, sigma,  which define the state of the fluid
'''

from fenics import *
import importlib
import ufl as ufl

import calculus as cal
import function_spaces as fsp
import numpy as np
import read_parameters as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

dt = rpam.T / rpam.num_steps  # time step size

print(f"xxx T = {rpam.T}")


# trial analytical expression for a vector
class TangentVelocityExpression(UserExpression):
    def eval(self, values, x):
        values[0] = -1.0
        values[1] = -x[1]**2

    def value_shape(self):
        return (2,)


# trial analytical expression for the  surface tension sigma(x,y)
class SurfaceTensionExpression(UserExpression):
    def eval(self, values, x):
        values[0] = x[1]**2

    def value_shape(self):
        return (1,)

'''

v__profile_l = Expression((f'4.0*1.5*x[1]*({rmsh.h} - x[1]) / pow({rmsh.h}, 2)', '0'), degree=2, h=rmsh.h)

bc_v__inflow = DirichletBC(fsp.Q_v, v__profile_l, rmsh.boundary_l)
bc_v__walls = DirichletBC(fsp.Q_v, Constant((0, 0)), rmsh.boundary_tb)
bc_v__cylinder = DirichletBC(fsp.Q_v, Constant((0, 0)), rmsh.boundary_circle)

bc_phi_outflow = DirichletBC(fsp.Q_phi, Constant(0), rmsh.boundary_r)

# boundary conditions for the surface_tension p
bc_v_ = [bc_v__walls, bc_v__inflow, bc_v__cylinder]
bc_phi = [bc_phi_outflow]

# Define variational problem for step 1
# step 1 for v_
F_v_ = ( \
                   rpam.rho * ((fsp.v_[i] - fsp.v_n_1[i]) / dt \
                          + (3.0 / 2.0 * fsp.v_n_1[j] - 1.0 / 2.0 * fsp.v_n_2[j]) * (fsp.V[i]).dx(j)) * fsp.nu_v_n[i] \
                   + fsp.sigma_n_32 * (fsp.nu_v_n[i]).dx(i) + rpam.mu * ((fsp.V[i]).dx(j) + (fsp.V[j]).dx(i)) * (fsp.nu_v_n[j]).dx(i) \
           ) * rmsh.dx

# step 2 for phi
F_phi = ((fsp.phi.dx(i)) * (fsp.nu_phi.dx(i)) + (rpam.rho / dt) * ((fsp.v_)[i].dx(i)) * fsp.nu_phi) * rmsh.dx

# step 3 for v_n
F_v_n = (((fsp.v_n[i] - fsp.v_[i]) + (dt / rpam.rho) * (fsp.phi.dx(i))) * fsp.nu_v_n[i]) * rmsh.dx
'''