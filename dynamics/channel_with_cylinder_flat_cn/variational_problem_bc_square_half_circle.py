from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size


# trial analytical expression for a vector
class TangentVelocityExpression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)


# trial analytical expression for the  surface tension sigma(x,y)
class SurfaceTensionExpression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0

    def value_shape(self):
        return (1,)


v__profile_l = Expression((f'4.0*1.5*x[1]*({2*rmsh.parameters["h"]} - x[1]) / pow({2*rmsh.parameters["h"]}, 2)', '0'), degree=2, h=rmsh.parameters["h"])



# the velocity on the inflow is fixed to (one half of) a Poiseille flow profile
bc_v__l = DirichletBC(fsp.Q_v, v__profile_l, rmsh.mf, rmsh.parameters['line_l_id'])
# on the bottom edge there is a wall: the velocity is zero there
bc_v__b = DirichletBC(fsp.Q_v, Constant((0, 0)), rmsh.mf, rmsh.parameters['line_b_id'])
# on the half circle there is a wall: the velcoty is zero there
bc_v__half_circle = DirichletBC(fsp.Q_v, Constant((0, 0)), rmsh.mf, rmsh.parameters['half_circle_id'])

# phi vanishes at the r edge because there sigma == 0 
bc_phi_r = DirichletBC(fsp.Q, Constant(0), rmsh.mf, rmsh.parameters['line_r_id'])

# boundary conditions for the surface_tension p
bc_v_ = [bc_v__l, bc_v__b, bc_v__half_circle]
bc_phi = [bc_phi_r]

# Define variational problem for step 1
# step 1 for v
F1 = ( \
                rpam.parameters['rho'] * ((fsp.v_[i] - fsp.v_n_1[i]) / dt \
                + (3.0 / 2.0 * fsp.v_n_1[j] - 1.0 / 2.0 * fsp.v_n_2[j]) * (fsp.V[i]).dx(j)) * fsp.nu[i] \
                + fsp.sigma_n_32 * (fsp.nu[i]).dx(i) \
                + rpam.parameters['mu'] * ((fsp.V[i]).dx(j) + (fsp.V[j]).dx(i)) * (fsp.nu[j]).dx(i) \
         ) * rmsh.dx

# step 2
F2 = ((fsp.phi.dx(i)) * (fsp.q.dx(i)) + (rpam.parameters['rho'] / dt) * ((fsp.v_)[i].dx(i)) * fsp.q) * rmsh.dx

# Define variational problem for step 3
F3 = (((fsp.v_n[i] - fsp.v_[i]) + (dt / rpam.parameters['rho']) * (fsp.phi.dx(i))) * fsp.nu[i]) * rmsh.dx
