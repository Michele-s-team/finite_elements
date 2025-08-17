from fenics import *
import importlib
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import read_parameters_solve as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices( 4 )


'''
if you compare with the solution from check-with-analytical-solution-bc-square-no-circle-a.nb:
    - z_t(b)_const_{here} <-> \[Phi]TOP(BOTTOM)_{check-with-analytical-solution-bc-square-no-circle.nb}
    - omega_t(b)_const_{here} <-> \[Psi]TOP(BOTTOM)_{check-with-analytical-solution-bc-square-no-circle.nb}

'''


class SurfaceTensionExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 1.0

    def value_shape(self):
        return (1,)


class z_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class omega_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


class mu_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0


    def value_shape(self):
        return (1,)

class tau_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0


    def value_shape(self):
        return (1,)


class n_omega_l_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)

class n_omega_r_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class n_omega_t_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters['n_omega_t_const']

    def value_shape(self):
        return (1,)

class n_omega_b_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters['n_omega_b_const']

    def value_shape(self):
        return (1,)


# the values of \partial_i z = omega_i on the circle and on the square, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
n_omega_l = interpolate(n_omega_l_Expression(element=fsp.Q_psi.ufl_element()), fsp.Q_psi)
n_omega_r = interpolate(n_omega_r_Expression(element=fsp.Q_psi.ufl_element()), fsp.Q_psi)
n_omega_t = interpolate(n_omega_t_Expression(element=fsp.Q_psi.ufl_element()), fsp.Q_psi)
n_omega_b = interpolate(n_omega_b_Expression(element=fsp.Q_psi.ufl_element()), fsp.Q_psi)


fsp.sigma.interpolate( SurfaceTensionExpression( element=fsp.Q_sigma.ufl_element() ) )
fsp.psi_0.interpolate(z_exact_Expression(element=fsp.Q_psi.ufl_element()))
fsp.rho_0.interpolate(omega_exact_Expression(element=fsp.Q_rho.ufl_element()))
fsp.zeta_0.interpolate(mu_exact_Expression(element=fsp.Q_zeta.ufl_element()))

fsp.tau_0.interpolate( tau_exact_Expression( element=fsp.Q_tau.ufl_element() ) )

fsp.psi_exact.interpolate(z_exact_Expression(element=fsp.Q_psi.ufl_element()))
fsp.rho_exact.interpolate(omega_exact_Expression(element=fsp.Q_rho.ufl_element()))
fsp.zeta_exact.interpolate(mu_exact_Expression(element=fsp.Q_zeta.ufl_element()))

fsp.tau_exact.interpolate( tau_exact_Expression( element=fsp.Q_tau.ufl_element() ) )

#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
fsp.assigner.assign(fsp.phi, [fsp.psi_0, fsp.rho_0, fsp.zeta_0])

# boundary conditions (BCs)

# BCs for z
#note that here I imposte BCs for z only on l and r because the solution is independent of x, if you consider cases where the solutoion depdends on x, add bc_l, bc_r
bc_z_t = DirichletBC( fsp.Q.sub( 0 ), rpam.parameters['z_t_const'], rmsh.boundary_t )
bc_z_b = DirichletBC( fsp.Q.sub( 0 ), rpam.parameters['z_b_const'], rmsh.boundary_b )

# all BCs
bcs = [bc_z_t, bc_z_b]

# Define variational problem

F_z = (rpam.parameters["kappa"] * (geo.g_c(fsp.rho)[i, j] * (fsp.zeta.dx(j)) * (fsp.nu_psi.dx(i)) - 2.0 * fsp.zeta * ((fsp.zeta) ** 2 - geo.K(fsp.rho)) * fsp.nu_psi) + fsp.sigma * fsp.zeta * fsp.nu_psi) * geo.sqrt_detg(fsp.rho) * rmsh.dx \
      - ( \
         # natural BC (bgeo.n_tb(fsp.omega))[i] * (fsp.mu.dx(i)) = 0 on ds_l and ds_r is imposed here
              (rpam.parameters["kappa"] * (bgeo.n_tb(fsp.rho))[i] * fsp.nu_psi * (fsp.zeta.dx(i))) * bgeo.sqrt_deth_tb(fsp.rho) * (rmsh.ds_t + rmsh.ds_b) \
      )

F_omega = (- fsp.psi * geo.Nabla_v(fsp.nu_rho, fsp.rho)[i, i] - fsp.rho[i] * fsp.nu_rho[i]) * geo.sqrt_detg(fsp.rho) * rmsh.dx \
          + ((bgeo.n_lr(fsp.rho))[i] * geo.g(fsp.rho)[i, j] * fsp.psi * fsp.nu_rho[j]) * bgeo.sqrt_deth_lr(fsp.rho) * (rmsh.ds_l + rmsh.ds_r) \
          + ((bgeo.n_tb(fsp.rho))[i] * geo.g(fsp.rho)[i, j] * fsp.psi * fsp.nu_rho[j]) * bgeo.sqrt_deth_tb(fsp.rho) * (rmsh.ds_t + rmsh.ds_b) \


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
F = ( F_z + F_omega + F_mu) + F_N

import variational_problem_pp_square_no_circle as vp_pp