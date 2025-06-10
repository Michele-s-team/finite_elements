'''
this module solves for the fields, v^n, sigma,  which define the state of the fluid
'''

from fenics import *
import importlib
import ufl as ufl

import calculus as cal
import elasticity as ela
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
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


# trial analytical expression for the  surface tension sigma(x,y)
class SurfaceTensionExpression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


v__profile_l = Expression((f'4.0*1.5*x[1]*({rmsh.h} - x[1]) / pow({rmsh.h}, 2)', '0'), element=fsp.Q_v_.ufl_element(), h=rmsh.h)
bc_v__l = DirichletBC(fsp.Q_v_, v__profile_l, rmsh.boundary_l)
bc_v__walls = DirichletBC(fsp.Q_v, Constant((0, 0)), rmsh.boundary_tb)

v__profile_ellipse = Expression((f'{fsp.omega_n} * (-sin({fsp.theta_n}) * (x[0] - {rmsh.focus[0]}) - cos({fsp.theta_n}) * (x[1] - {rmsh.focus[1]}))', f'{fsp.omega_n} * (cos({fsp.theta_n}) * (x[0] - {rmsh.focus[0]}) - sin({fsp.theta_n}) * (x[1] - {rmsh.focus[1]}))'), element=fsp.Q_v_.ufl_element())
bc_v__ellipse = DirichletBC(fsp.Q_v_, v__profile_ellipse, rmsh.boundary_ellipse)

bc_phi_r = DirichletBC(fsp.Q_phi, Constant(0), rmsh.boundary_r)

# boundary conditions for the surface_tension p
bc_v_ = [bc_v__l, bc_v__walls, bc_v__ellipse]
bc_phi = [bc_phi_r]

# sign


# Define variational problem for step 1
# step 1 for v_
F_v_ = ( \
                   rpam.rho * ((fsp.v_[i] - fsp.v_n_1[i]) / dt \
                               + (3.0 / 2.0 * (fsp.v_n_1[k] - fsp.u_dot_n_1[k]) * ela.G(fsp.u_n_1)[j, k] - 1.0 / 2.0 * (fsp.v_n_2[k] - fsp.u_dot_n_2[k]) * ela.G(fsp.u_n_2)[j, k]) * (fsp.V[i]).dx(j)) * fsp.nu_v_[i] \
                   + fsp.sigma_n_32 * ela.G(fsp.u_n_1)[l, i] * (fsp.nu_v_[i]).dx(l) + rpam.mu * ela.G(fsp.u_n_1)[k, j] * ((fsp.V[i]).dx(k)) * ela.G(fsp.u_n_1)[l, j] * (fsp.nu_v_[i]).dx(l) \
           ) * ela.detF(fsp.u_n_1) * rmsh.dx

# step 2 for phi
F_phi = ((fsp.phi.dx(i)) * (fsp.nu_phi.dx(i)) + (rpam.rho / dt) * ((fsp.v_)[i].dx(i)) * fsp.nu_phi) * rmsh.dx

# step 3 for v_n
F_v_n = (((fsp.v_n[i] - fsp.v_[i]) + (dt / rpam.rho) * (fsp.phi.dx(i))) * fsp.nu_v_n[i]) * rmsh.dx
