from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import differential_geometry.boundary.geometry as bgeo
import fluid as flu
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

alpha, beta, gamma, delta = ufl.indices(4)


dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size

class v0_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)
    
class sigma0_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0

    def value_shape(self):
        return (1,)


class f_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)

class tau_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)

fsp.f.interpolate(f_Expression(element=fsp.Q_f.ufl_element()))
fsp.tau.interpolate(tau_Expression(element=fsp.Q_tau.ufl_element()))

# step 1
F1 = ( \
                rpam.parameters['rho'] * (
                    (fsp.v_[alpha] - fsp.v_n_1[alpha]) / dt \
                    + (3.0 / 2.0 * fsp.v_n_1[beta] - 1.0 / 2.0 * fsp.v_n_2[beta]) * (fsp.V[alpha]).dx(beta) - fsp.f[alpha]
                    ) * fsp.nu_v_[alpha] \
                +  flu.sigma(fsp.V, fsp.sigma_n_32, rpam.parameters['mu'])[alpha, beta] * fsp.nu_v_[alpha].dx(beta)
      ) * rmsh.dx \
      - (fsp.tau[alpha] * fsp.nu_v_[alpha]) * rmsh.ds

# step 2
F2_phi = ((fsp.phi.dx(alpha)) * (fsp.nu_phi.dx(alpha)) + (rpam.parameters['rho'] / dt) * ((fsp.v_)[alpha].dx(alpha)) * fsp.nu_phi) * rmsh.dx \
        - ( bgeo.facet_normal[alpha] * (fsp.phi.dx(alpha)) * fsp.nu_phi) * rmsh.ds
F2_omega = (fsp.omega[alpha] - fsp.phi.dx(alpha)) * (fsp.nu_omega[alpha] - fsp.nu_phi.dx(alpha)) * rmsh.dx

F2_N = rpam.parameters['alpha'] / rmsh.r_mesh * (
    fsp.phi - rpam.parameters['mu'] * bgeo.facet_normal[alpha] * bgeo.facet_normal[beta] * (fsp.v_[alpha] - fsp.v_n_2[alpha] - dt/rpam.parameters['rho'] * fsp.omega[alpha]).dx(beta)) * \
    (
        fsp.nu_phi + rpam.parameters['mu'] * dt / rpam.parameters['rho'] *  bgeo.facet_normal[gamma] * bgeo.facet_normal[delta] * fsp.nu_omega[gamma].dx(delta)
    ) * rmsh.ds

F2 = (F2_phi + F2_omega) + F2_N

# step 3
F3 = (((fsp.v_n[alpha] - fsp.v_[alpha]) + (dt / rpam.parameters['rho']) * (fsp.phi.dx(alpha))) * fsp.nu_v_n[alpha]) * rmsh.dx
