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

F_z = (rpam.parameters["kappa"] * (geo.g_c(fsp.rho)[i, j] * (fsp.zeta.dx(j)) * (fsp.nu_psi.dx(i)) - 2.0 * fsp.zeta * ((fsp.zeta) ** 2 - geo.K(fsp.rho)) * fsp.nu_psi) + fsp.sigma * fsp.zeta * fsp.nu_psi) * geo.sqrt_detg(fsp.rho) * rmsh.dx \
      - ( \
          # natural BC (bgeo.n_tb(fsp.omega))[i] * (fsp.mu.dx(i)) = 0 on ds_l and ds_r is imposed here
              (rpam.parameters["kappa"] * (bgeo.n_tb(fsp.rho))[i] * fsp.nu_psi * (fsp.zeta.dx(i))) * bgeo.sqrt_deth_tb(fsp.rho) * (rmsh.ds_t + rmsh.ds_b) \
          )

F_omega = (- fsp.psi * geo.Nabla_v(fsp.nu_rho, fsp.rho)[i, i] - fsp.rho[i] * fsp.nu_rho[i]) * geo.sqrt_detg(fsp.rho) * rmsh.dx \
          + ((bgeo.n_lr(fsp.rho))[i] * geo.g(fsp.rho)[i, j] * fsp.psi * fsp.nu_rho[j]) * bgeo.sqrt_deth_lr(fsp.rho) * (rmsh.ds_l + rmsh.ds_r) \
          + ((bgeo.n_tb(fsp.rho))[i] * geo.g(fsp.rho)[i, j] * fsp.psi * fsp.nu_rho[j]) * bgeo.sqrt_deth_tb(fsp.rho) * (rmsh.ds_t + rmsh.ds_b) \
 \
    F_mu = ((geo.H(fsp.rho) - fsp.zeta) * fsp.nu_zeta) * geo.sqrt_detg(fsp.rho) * rmsh.dx

F_N = rpam.parameters["alpha"] / rmsh.r_mesh * ( \
            + (((bgeo.n_lr(fsp.rho))[i] * fsp.rho[i] - n_omega_l) * ((bgeo.n_lr(fsp.rho))[k] * geo.g(fsp.rho)[k, l] * fsp.nu_rho[l])) * bgeo.sqrt_deth_lr(fsp.rho) * rmsh.ds_l \
            + (((bgeo.n_lr(fsp.rho))[i] * fsp.rho[i] - n_omega_r) * ((bgeo.n_lr(fsp.rho))[k] * geo.g(fsp.rho)[k, l] * fsp.nu_rho[l])) * bgeo.sqrt_deth_lr(fsp.rho) * rmsh.ds_r \
            + (((bgeo.n_tb(fsp.rho))[i] * fsp.rho[i] - n_omega_t) * ((bgeo.n_tb(fsp.rho))[k] * geo.g(fsp.rho)[k, l] * fsp.nu_rho[l])) * bgeo.sqrt_deth_tb(fsp.rho) * rmsh.ds_t \
            + (((bgeo.n_tb(fsp.rho))[i] * fsp.rho[i] - n_omega_b) * ((bgeo.n_tb(fsp.rho))[k] * geo.g(fsp.rho)[k, l] * fsp.nu_rho[l])) * bgeo.sqrt_deth_tb(fsp.rho) * rmsh.ds_b \
            # these terms constrain mu = H(omega) on the boundary
            + ((geo.H(fsp.rho) - fsp.zeta) * fsp.nu_zeta) * bgeo.sqrt_deth_lr(fsp.rho) * rmsh.ds_lr \
            + ((geo.H(fsp.rho) - fsp.zeta) * fsp.nu_zeta) * bgeo.sqrt_deth_tb(fsp.rho) * rmsh.ds_tb \
    )

# total functional for the mixed problem
F = (F_z + F_omega + F_mu) + F_N

import variational_problem_pp_square_no_circle as vp_pp
