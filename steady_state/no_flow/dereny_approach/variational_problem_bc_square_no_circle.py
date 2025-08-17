from fenics import *
import importlib
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import read_parameters_solve as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)


class sigma_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters['sigma_const']

    def value_shape(self):
        return (1,)


class psi_exact_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class omega_exact_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class rho_exact_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class zeta_exact_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


fsp.sigma.interpolate(sigma_Expression(element=fsp.Q_sigma.ufl_element()))

fsp.psi_0.interpolate(psi_exact_Expression(element=fsp.Q_psi.ufl_element()))
fsp.omega_0.interpolate(omega_exact_Expression(element=fsp.Q_omega.ufl_element()))
fsp.rho_0.interpolate(rho_exact_Expression(element=fsp.Q_rho.ufl_element()))
fsp.zeta_0.interpolate(zeta_exact_Expression(element=fsp.Q_zeta.ufl_element()))

fsp.psi_exact.interpolate(psi_exact_Expression(element=fsp.Q_psi.ufl_element()))
fsp.omega_exact.interpolate(omega_exact_Expression(element=fsp.Q_omega.ufl_element()))
fsp.rho_exact.interpolate(rho_exact_Expression(element=fsp.Q_rho.ufl_element()))
fsp.zeta_exact.interpolate(zeta_exact_Expression(element=fsp.Q_zeta.ufl_element()))

# uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
# fsp.assigner.assign(fsp.phi, [fsp.psi_0, fsp.omega_0, fsp.rho_0, fsp.zeta_0])


bc_psi_l = DirichletBC(fsp.Q.sub(0), rpam.parameters['psi_l'], rmsh.boundary_l)
bc_psi_r = DirichletBC(fsp.Q.sub(0), rpam.parameters['psi_r'], rmsh.boundary_r)

bc_omega_l = DirichletBC(fsp.Q.sub(1), rpam.parameters['omega_l'], rmsh.boundary_l)

bc_rho_l = DirichletBC(fsp.Q.sub(2), rpam.parameters['rho_l'], rmsh.boundary_l)
bc_zeta_l = DirichletBC(fsp.Q.sub(3), rpam.parameters['zeta_l'], rmsh.boundary_l)

bcs = [bc_psi_l, bc_psi_r, bc_omega_l, bc_rho_l, bc_zeta_l]

# Define variational problem

F_psi = ( \
                    ( \
                                rpam.parameters['kappa'] * ( \
                                    1.0 / (8.0 * (fsp.rho) ** 3) * (5.0 * sin(fsp.psi) + sin(3 * fsp.psi)) - \
                                    1.0 / (4.0 * (fsp.rho) ** 2) * fsp.omega * (1.0 + 3.0 * cos(2.0 * fsp.psi)) - \
                                    3.0 / (2.0 * fsp.rho) * sin(fsp.psi) * (fsp.omega) ** 2 + \
                                    1.0 / 2.0 * (fsp.omega) ** 2 + \
                                    2.0 / (fsp.rho) * cos(fsp.psi) * fsp.omega.dx(0) \
                            ) - \
                                fsp.sigma * (1.0 / (fsp.rho) * sin(fsp.psi) + fsp.omega) \
                        ) * fsp.nu_psi - \
                    rpam.parameters['kappa'] * fsp.omega.dx(0) * fsp.nu_psi.dx(0)
        ) * rmsh.dx + \
        (rpam.parameters['kappa'] * bgeo.facet_normal[0] * fsp.omega.dx(0) * fsp.nu_psi) * rmsh.ds
'''
omega = \partial_1 psi
<omega nu_omega>_Omega = <(\partial_1 psi) nu_omega>_Omega
                       = <(\partial_i psi) \delta_{i1} nu_omega>_Omega
                       = - <psi \partial_1 nu_omega>_Omega + <n_1 psi nu_omega>_{\partial Omega}
                       
<omega nu_omega + psi \partial_1 nu_omega>_Omega - <n_1 psi nu_omega>_{\partial Omega} = 0                
'''
F_omega = (fsp.omega * fsp.nu_omega + fsp.psi * fsp.nu_omega.dx(0)) * rmsh.dx - \
          (bgeo.facet_normal[0] * fsp.psi * fsp.nu_omega) * rmsh.ds

'''
\partial_1 rho = cos(psi)
<(\partial_1 rho) nu_rho >_Omega = <cos(psi) nu_rho>_Omega
<(\partial_i rho)  \delta_{i1} nu_rho >_Omega = <cos(psi) nu_rho>_Omega
-< rho  \partial_1 nu_rho >_Omega + <n_1  rho nu_rho >_{\partial Omega} = <cos(psi) nu_rho>_Omega 

<cos(psi) nu_rho + rho  \partial_1 nu_rho >_Omega  - <n_1  rho nu_rho>_{\partial Omega} = 0
'''

F_rho = (cos(fsp.psi) * fsp.nu_rho + fsp.rho * fsp.nu_rho.dx(0)) * rmsh.dx - \
        - (bgeo.facet_normal[0] * fsp.rho * fsp.nu_rho) * rmsh.ds

F_zeta = (-sin(fsp.psi) * fsp.nu_zeta + fsp.zeta * fsp.nu_zeta.dx(0)) * rmsh.dx - \
         - (bgeo.facet_normal[0] * fsp.zeta * fsp.nu_zeta) * rmsh.ds

'''
F_N = rpam.parameters["alpha"] / rmsh.r_mesh * ( \
        + () * bgeo.sqrt_deth_lr(fsp.rho) * rmsh.ds_lr \
    )
    

'''
# total functional for the mixed problem
F = (F_psi + F_omega + F_rho + F_zeta)
