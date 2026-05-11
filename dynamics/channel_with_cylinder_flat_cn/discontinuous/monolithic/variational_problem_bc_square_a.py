from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import physics.fluid_mechanics as flu
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

alpha, beta, gamma, delta = ufl.indices(4)


dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size

class v_l_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rpam.parameters['v__l_const'] * 4.0 * 1.5 * x[1] * (rmsh.parameters['h'] - x[1]) / (rmsh.parameters['h']**2)
        values[1] = 0.0

    def value_shape(self):
        return (2,)
    
class v_tb_circle_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


class f_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)

class tau_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)

msh.interpolate_dg(fsp.v_l, v_l_expression())
msh.interpolate_dg(fsp.v_tb_circle, v_tb_circle_expression())

msh.interpolate_dg(fsp.f, f_expression())
msh.interpolate_dg(fsp.tau, tau_expression())

bcs_psi = [] 


# step 1
F_v_n = ( \
                ( \
                    rpam.parameters['rho'] * ( (fsp.v_n[alpha] - fsp.v_n_1[alpha]) / dt + fsp.v_n[beta]  * (fsp.v_n[alpha]).dx(beta) ) - fsp.f[alpha] \
                ) * fsp.nu_v_n[alpha] \
                + flu.sigma(fsp.v_n, fsp.sigma_n, rpam.parameters['mu'])[alpha, beta] * fsp.nu_v_n[alpha].dx(beta) \
      ) * rmsh.dx \
      - (msh.jump(fsp.nu_v_n[beta], bgeo.facet_normal)[alpha] * msh.average(flu.sigma(fsp.v_n, fsp.sigma_n, rpam.parameters['mu'])[beta, alpha])) * rmsh.dS \
      - (bgeo.facet_normal[alpha] * flu.sigma(fsp.v_n, fsp.sigma_n, rpam.parameters['mu'])[beta, alpha] * fsp.nu_v_n[beta]) * (rmsh.ds_l + rmsh.ds_tb + rmsh.ds_circle) \
      - (fsp.tau[beta] * fsp.nu_v_n[beta]) * rmsh.ds_r

F_N = rpam.parameters['alpha']/rmsh.r_mesh * (\
            msh.jump(fsp.v_n[alpha], bgeo.facet_normal)[beta] * msh.jump(fsp.nu_v_n[alpha], bgeo.facet_normal)[beta] * rmsh.dS + \
            (fsp.v_n[alpha] - fsp.v_l[alpha]) * fsp.nu_v_n[alpha] * rmsh.ds_l + \
            (fsp.v_n[alpha] - fsp.v_tb_circle[alpha]) * fsp.nu_v_n[alpha] * (rmsh.ds_tb + rmsh.ds_circle) + \
            (fsp.sigma_n - fsp.sigma_r) * fsp.nu_sigma_n * rmsh.ds_r \
        )

F_sigma_n = ( fsp.v_n[alpha].dx(alpha) * fsp.nu_sigma_n ) * rmsh.dx


F = F_v_n + F_sigma_n