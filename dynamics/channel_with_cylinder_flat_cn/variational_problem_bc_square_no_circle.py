from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
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
        # values[0] = 4*x[0]*x[1]*sin(8*(norm(np.subtract(x, c_r)) - r))*sin(8*(norm(np.subtract(x, c_R)) - R))
        # values[0] = cos(norm(np.subtract(x, c_r)) - r) * sin(norm(np.subtract(x, c_R)) - R)
        values[0] = 0.0

    def value_shape(self):
        return (1,)


v__profile_l = Expression((f'{rpam.parameters["v__l_const"]} * 4.0*1.5*x[1]*({rmsh.parameters["h"]} - x[1])/ pow({rmsh.parameters["h"]}, 2)', '0'), degree=2, h=rmsh.parameters["h"])

bc_v__inflow = DirichletBC(fsp.Q_v, v__profile_l, rmsh.boundary_l)
bc_v__walls = DirichletBC(fsp.Q_v, Constant((0, 0)), rmsh.boundary_tb)

bc_phi_outflow = DirichletBC(fsp.Q, Constant(0), rmsh.boundary_r)

# boundary conditions for the surface_tension p
bc_v_ = [bc_v__walls, bc_v__inflow]
bc_phi = [bc_phi_outflow]

# Define variational problem for step 1
# step 1 for v
F1 = ( \
                 rpam.parameters['rho'] * ((fsp.v_[i] - fsp.v_n_1[i]) / dt \
                        + (3.0 / 2.0 * fsp.v_n_1[j] - 1.0 / 2.0 * fsp.v_n_2[j]) * (fsp.V[i]).dx(j)) * fsp.nu[i] \
                 + fsp.sigma_n_32 * (fsp.nu[i]).dx(i) + rpam.parameters['mu'] * ((fsp.V[i]).dx(j) + (fsp.V[j]).dx(i)) * (fsp.nu[j]).dx(i) \
         ) * rmsh.dx

# step 2
F2 = ((fsp.phi.dx(i)) * (fsp.q.dx(i)) + (rpam.parameters['rho'] / dt) * ((fsp.v_)[i].dx(i)) * fsp.q) * rmsh.dx

# Define variational problem for step 3
F3 = (((fsp.v_n[i] - fsp.v_[i]) + (dt / rpam.parameters['rho']) * (fsp.phi.dx(i))) * fsp.nu[i]) * rmsh.dx