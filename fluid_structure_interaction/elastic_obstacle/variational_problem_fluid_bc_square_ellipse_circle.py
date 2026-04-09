'''
this module solves for the fields, v^n, sigma,  which define the state of the fluid
'''

from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size


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
        values[0] = rpam.parameters['sigma_r']

    def value_shape(self):
        return (1,)


v__profile_l = Expression((f'{rpam.parameters["v_l"]}* 4.0*1.5*x[1]*({rmsh.parameters["h"]} - x[1]) / pow({rmsh.parameters["h"]}, 2)', '0'), element=fsp.Q_v_.ufl_element())

bc_v__l = DirichletBC(fsp.Q_v_, v__profile_l, rmsh.boundary[1]['l'])
bc_v__tb = DirichletBC(fsp.Q_v_, Constant((0, 0)), rmsh.boundary[1]['tb'])
bc_v__ellipse = DirichletBC(fsp.Q_v_, fsp.u_msh_dot_n, rmsh.boundary[1]['ellipse'])
bc_v_ = [bc_v__l, bc_v__tb, bc_v__ellipse]

bc_phi_r = DirichletBC(fsp.Q_phi, Constant(0), rmsh.boundary[1]['r'])
bc_phi = [bc_phi_r]


# Define variational problem for step 1
# step 1 for v_
# natural BC imposed here
F_v_ = ( \
                   rpam.parameters['rho_fluid'] * ((fsp.v_[i] - fsp.v_n_1[i]) / dt \
                               + (3.0 / 2.0 * (fsp.v_n_1[k] - fsp.u_msh_dot_n_1[k]) * ela.G(fsp.u_msh_n_1)[j, k] - 1.0 / 2.0 * (fsp.v_n_2[k] - fsp.u_msh_dot_n_2[k]) * ela.G(fsp.u_msh_n_2)[j, k]) * (fsp.V[i]).dx(j)) * fsp.nu_v_[i] \
                   + fsp.sigma_n_32 * ela.G(fsp.u_msh_n_1)[l, i] * (fsp.nu_v_[i]).dx(l) + rpam.parameters['mu_fluid'] * ela.G(fsp.u_msh_n_1)[k, j] * ((fsp.V[i]).dx(k)) * ela.G(fsp.u_msh_n_1)[l, j] * (fsp.nu_v_[i]).dx(l) \
           ) * ela.detF(fsp.u_msh_n_1) * rmsh.dx_sub_mesh[1] \
       - (ela.G(fsp.u_msh_n_1)[l, i] * bgeo.sub_mesh_facet_normal[1][l] * fsp.sigma_n_32 * fsp.nu_v_[i]) * ela.detF(fsp.u_msh_n_1) * rmsh.ds_sub_mesh[1]['ds']  \
       - ( \
                   rpam.parameters['mu_fluid'] * ela.G(fsp.u_msh_n_1)[l, j] * bgeo.sub_mesh_facet_normal[1][l] * ela.G(fsp.u_msh_n_1)[k, j] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_msh_n_1) * rmsh.ds_sub_mesh[1]['ds_l']\
                   + rpam.parameters['mu_fluid'] * ela.G(fsp.u_msh_n_1)[l, j] * bgeo.sub_mesh_facet_normal[1][l] * ela.G(fsp.u_msh_n_1)[k, j] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_msh_n_1) * rmsh.ds_sub_mesh[1]['ds_tb'] \
                   + rpam.parameters['mu_fluid'] * ela.G(fsp.u_msh_n_1)[l, j] * bgeo.sub_mesh_facet_normal[1][l] * ela.G(fsp.u_msh_n_1)[k, j] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_msh_n_1) * rmsh.ds_sub_mesh[1]['ds_ellipse'] \
                   + rpam.parameters['mu_fluid'] * ela.G(fsp.u_msh_n_1)[l, 1] * bgeo.sub_mesh_facet_normal[1][l] * ela.G(fsp.u_msh_n_1)[k, 1] * (fsp.V[i].dx(k)) * fsp.nu_v_[i] * ela.detF(fsp.u_msh_n_1) * rmsh.ds_sub_mesh[1]['ds_r'] \
           )
# step 2 for phi
F_phi = ( \
                    - ela.G(fsp.u_msh_n_1)[j, i] * (fsp.phi.dx(j)) * ela.G(fsp.u_msh_n_1)[l, i] * (fsp.nu_phi.dx(l)) \
                    - (rpam.parameters['rho_fluid'] / dt) * ela.G(fsp.u_msh_n_1)[j, i] * ((fsp.v_[i]).dx(j)) * fsp.nu_phi \
            ) * ela.detF(fsp.u_msh_n_1) * rmsh.dx_sub_mesh[1] \
        + (ela.G(fsp.u_msh_n_1)[l, i] * bgeo.sub_mesh_facet_normal[1][l] * ela.G(fsp.u_msh_n_1)[j, i] * (fsp.phi.dx(j)) * fsp.nu_phi) * ela.detF(fsp.u_msh_n_1) * rmsh.ds_sub_mesh[1]['ds_r']


# step 3 for v_n
F_v_n = ( ( (fsp.v_n[i] - fsp.v_[i]) + (dt / rpam.parameters['rho_fluid']) * ela.G(fsp.u_msh_n_1)[l, i] * (fsp.phi.dx(l)) ) * fsp.nu_v_n[i] ) * ela.detF(fsp.u_msh_n_1) * rmsh.dx_sub_mesh[1]
