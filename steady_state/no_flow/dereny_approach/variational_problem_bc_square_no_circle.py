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
fsp.zeta_exact.interpolat(zeta_exact_Expression(element=fsp.Q_zeta.ufl_element()))

# uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
# fsp.assigner.assign(fsp.phi, [fsp.psi_0, fsp.omega_0, fsp.rho_0, fsp.zeta_0])


bc_psi_l = DirichletBC(fsp.Q.sub(0), rpam.parameters['psi_l'], rmsh.boundary_l)
bc_psi_r = DirichletBC(fsp.Q.sub(0), rpam.parameters['psi_r'], rmsh.boundary_r)

bc_omega_l = DirichletBC(fsp.Q.sub(1), rpam.parameters['omega_l'], rmsh.boundary_l)

bc_rho_l = DirichletBC(fsp.Q.sub(2), rpam.parameters['rho_l'], rmsh.boundary_l)
bc_zeta_l = DirichletBC(fsp.Q.sub(3), rpam.parameters['zeta_l'], rmsh.boundary_l)

bcs = [bc_psi_l, bc_psi_r, bc_omega_l, bc_rho_l, bc_zeta_l]

# Define variational problem

F_psi = () * geo.sqrt_detg(fsp.rho) * rmsh.dx \
        - () * bgeo.sqrt_deth_tb(fsp.rho) * (rmsh.ds_t + rmsh.ds_b) \

F_omega = () * geo.sqrt_detg(fsp.rho) * rmsh.dx \
          + () * bgeo.sqrt_deth_lr(fsp.rho) * (rmsh.ds_l + rmsh.ds_r) \

F_mu = ((geo.H(fsp.rho) - fsp.zeta) * fsp.nu_zeta) * geo.sqrt_detg(fsp.rho) * rmsh.dx

F_N = rpam.parameters["alpha"] / rmsh.r_mesh * ( \
    # these terms constrain mu = H(omega) on the boundary
        + ((geo.H(fsp.rho) - fsp.zeta) * fsp.nu_zeta) * bgeo.sqrt_deth_lr(fsp.rho) * rmsh.ds_lr \
        + ((geo.H(fsp.rho) - fsp.zeta) * fsp.nu_zeta) * bgeo.sqrt_deth_tb(fsp.rho) * rmsh.ds_tb \
    )

# total functional for the mixed problem
F = (F_psi + F_omega + F_mu) + F_N
