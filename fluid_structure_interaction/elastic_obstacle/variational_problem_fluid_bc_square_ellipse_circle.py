'''
this module solves for the fields, v^n, sigma,  which define the state of the fluid
'''

from fenics import *
import importlib
import ufl as ufl

import boundary_geometry as bgeo
import elasticity as ela
import function_spaces as fsp
import read_parameters as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

dt = rpam.T / rpam.num_steps  # time step size


# trial analytical expression for a vector
class v_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


# trial analytical expression for the  surface tension sigma(x,y)
class sigma_expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.sigma_r

    def value_shape(self):
        return (1,)


v__profile_l = Expression((f'{rpam.v_l}* 4.0*1.5*x[1]*({rmsh.parameters["h"]} - x[1]) / pow({rmsh.parameters["h"]}, 2)', '0'), element=fsp.Q_v_.ufl_element())

bc_v__l = DirichletBC(fsp.Q_v_, v__profile_l, rmsh.boundary[1]['l'])
bc_v__tb = DirichletBC(fsp.Q_v_, Constant((0, 0)), rmsh.boundary[1]['tb'])
bc_v__ellipse = DirichletBC(fsp.Q_v_, fsp.u_el_dot_n_on_sub_mesh_1, rmsh.boundary[1]['ellipse'])
bc_v_ = [bc_v__l, bc_v__tb, bc_v__ellipse]

bc_phi_r = DirichletBC(fsp.Q_phi, Constant(0), rmsh.boundary[1]['r'])
bc_phi = [bc_phi_r]

'''
# Define variational problem for step 1
# step 1 for v_
F_v_ = ( \
                   rpam.rho_fluid * ((fsp.v_[i] - fsp.v_n_1[i]) / dt \
                               + (3.0 / 2.0 * (fsp.v_n_1[k] - fsp.u_el_dot_n_1[k]) * ela.G(fsp.u_el_n_1)[j, k] - 1.0 / 2.0 * (fsp.v_n_2[k] - fsp.u_el_dot_n_2[k]) * ela.G(fsp.u_el_n_2)[j, k]) * (fsp.V[i]).dx(j)) * fsp.nu_v_[i] \
                   + fsp.sigma_n_32 * ela.G(fsp.u_el_n_1)[l, i] * (fsp.nu_v_[i]).dx(l) + rpam.mu_fluid * ela.G(fsp.u_el_n_1)[k, j] * ((fsp.V[i]).dx(k)) * ela.G(fsp.u_el_n_1)[l, j] * (fsp.nu_v_[i]).dx(l) \
           ) * ela.detF(fsp.u_el_n_1) * rmsh.dx \
       - (ela.G(fsp.u_el_n_1)[l, i] * bgeo.facet_normal[l] * fsp.sigma_n_32 * fsp.nu_v_[i]) * ela.detF(fsp.u_el_n_1) * rmsh.ds \
       - ( \
                   rpam.mu_fluid * ela.G(fsp.u_el_n_1)[l, j] * bgeo.facet_normal[l] * ela.G(fsp.u_el_n_1)[k, j] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_el_n_1) * rmsh.ds_l \
                   + rpam.mu_fluid * ela.G(fsp.u_el_n_1)[l, j] * bgeo.facet_normal[l] * ela.G(fsp.u_el_n_1)[k, j] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_el_n_1) * rmsh.ds_tb \
                   + rpam.mu_fluid * ela.G(fsp.u_el_n_1)[l, j] * bgeo.facet_normal[l] * ela.G(fsp.u_el_n_1)[k, j] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_el_n_1) * rmsh.ds_ellipse \
                   + rpam.mu_fluid * ela.G(fsp.u_el_n_1)[l, 1] * bgeo.facet_normal[l] * ela.G(fsp.u_el_n_1)[k, 1] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_el_n_1) * rmsh.ds_r \
           )

# step 2 for phi
F_phi = ( \
                    - ela.G(fsp.u_el_n_1)[j, i] * (fsp.phi.dx(j)) * ela.G(fsp.u_el_n_1)[l, i] * (fsp.nu_phi.dx(l)) \
                    - (rpam.rho_fluid / dt) * ela.G(fsp.u_el_n_1)[j, i] * ((fsp.v_[i]).dx(j)) * fsp.nu_phi \
            ) * ela.detF(fsp.u_el_n_1) * rmsh.dx \
        + (ela.G(fsp.u_el_n_1)[l, i] * bgeo.facet_normal[l] * ela.G(fsp.u_el_n_1)[j, i] * (fsp.phi.dx(j)) * fsp.nu_phi) * ela.detF(fsp.u_el_n_1) * rmsh.ds_r

# step 3 for v_n
F_v_n = (((fsp.v_n[i] - fsp.v_[i]) + (dt / rpam.rho_fluid) * ela.G(fsp.u_el_n_1)[l, i] * (fsp.phi.dx(l))) * fsp.nu_v_n[i]) * ela.detF(fsp.u_el_n_1) * rmsh.dx
'''