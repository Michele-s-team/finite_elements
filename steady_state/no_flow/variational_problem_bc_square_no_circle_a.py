from fenics import *
import importlib
import ufl as ufl

import command as cmd
import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import read_parameters_solve as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

cmd.set_gauge('monge')

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
n_omega_l = interpolate(n_omega_l_Expression(element=fsp.Q_z.ufl_element()), fsp.Q_z)
n_omega_r = interpolate(n_omega_r_Expression(element=fsp.Q_z.ufl_element()), fsp.Q_z)
n_omega_t = interpolate(n_omega_t_Expression(element=fsp.Q_z.ufl_element()), fsp.Q_z)
n_omega_b = interpolate(n_omega_b_Expression(element=fsp.Q_z.ufl_element()), fsp.Q_z)


fsp.sigma.interpolate( SurfaceTensionExpression( element=fsp.Q_sigma.ufl_element() ) )
fsp.z_0.interpolate( z_exact_Expression( element=fsp.Q_z.ufl_element() ) )
fsp.omega_0.interpolate( omega_exact_Expression( element=fsp.Q_omega.ufl_element() ) )
fsp.mu_0.interpolate( mu_exact_Expression( element=fsp.Q_mu.ufl_element() ) )

fsp.tau_0.interpolate( tau_exact_Expression( element=fsp.Q_tau.ufl_element() ) )

fsp.z_exact.interpolate( z_exact_Expression( element=fsp.Q_z.ufl_element() ) )
fsp.omega_exact.interpolate( omega_exact_Expression( element=fsp.Q_omega.ufl_element() ) )
fsp.mu_exact.interpolate( mu_exact_Expression( element=fsp.Q_mu.ufl_element() ) )

fsp.tau_exact.interpolate( tau_exact_Expression( element=fsp.Q_tau.ufl_element() ) )

#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
fsp.assigner.assign(fsp.psi, [fsp.z_0, fsp.omega_0, fsp.mu_0])

# boundary conditions (BCs)

# BCs for z
#note that here I imposte BCs for z only on l and r because the solution is independent of x, if you consider cases where the solutoion depdends on x, add bc_l, bc_r
bc_z_t = DirichletBC( fsp.Q.sub( 0 ), rpam.parameters['z_t_const'], rmsh.boundary_t )
bc_z_b = DirichletBC( fsp.Q.sub( 0 ), rpam.parameters['z_b_const'], rmsh.boundary_b )

# all BCs
bcs = [bc_z_t, bc_z_b]

# Define variational problem

F_z = ( rpam.parameters["kappa"] * ( geo.g_c(fsp.omega)[i, j] * (fsp.mu.dx(j)) * (fsp.nu_z.dx(i)) - 2.0 * fsp.mu * ( (fsp.mu)**2 - geo.K(fsp.omega) ) * fsp.nu_z ) + fsp.sigma * fsp.mu * fsp.nu_z ) * geo.sqrt_detg(fsp.omega) * rmsh.dx \
    - ( \
         # natural BC (bgeo.n_tb(fsp.omega))[i] * (fsp.mu.dx(i)) = 0 on ds_l and ds_r is imposed here
         ( rpam.parameters["kappa"] * (bgeo.n_tb(fsp.omega))[i] * fsp.nu_z * (fsp.mu.dx(i)) ) * bgeo.sqrt_deth_tb(fsp.omega) * (rmsh.ds_t + rmsh.ds_b) \
      )

F_omega = ( - fsp.z * geo.Nabla_v(fsp.nu_omega, fsp.omega)[i, i] - fsp.omega[i] * fsp.nu_omega[i] ) *  geo.sqrt_detg(fsp.omega) * rmsh.dx \
          + ( (bgeo.n_lr(fsp.omega))[i] * geo.g(fsp.omega)[i, j] * fsp.z * fsp.nu_omega[j] ) * bgeo.sqrt_deth_lr(fsp.omega) * (rmsh.ds_l + rmsh.ds_r) \
          + ( (bgeo.n_tb(fsp.omega))[i] * geo.g(fsp.omega)[i, j] * fsp.z * fsp.nu_omega[j] ) * bgeo.sqrt_deth_tb(fsp.omega) * (rmsh.ds_t + rmsh.ds_b) \


F_mu = ((geo.H( fsp.omega ) - fsp.mu) * fsp.nu_mu) * geo.sqrt_detg( fsp.omega ) * rmsh.dx

F_N = rpam.parameters["alpha"] / rmsh.r_mesh * ( \
            + (((bgeo.n_lr(fsp.omega))[i] * fsp.omega[i] - n_omega_l) * ((bgeo.n_lr(fsp.omega))[k] * geo.g(fsp.omega)[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_lr(fsp.omega) * rmsh.ds_l \
            + (((bgeo.n_lr(fsp.omega))[i] * fsp.omega[i] - n_omega_r) * ((bgeo.n_lr(fsp.omega))[k] * geo.g(fsp.omega)[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_lr(fsp.omega) * rmsh.ds_r \
            + (((bgeo.n_tb(fsp.omega))[i] * fsp.omega[i] - n_omega_t) * ((bgeo.n_tb(fsp.omega))[k] * geo.g(fsp.omega)[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_tb(fsp.omega) * rmsh.ds_t \
            + (((bgeo.n_tb(fsp.omega))[i] * fsp.omega[i] - n_omega_b) * ((bgeo.n_tb(fsp.omega))[k] * geo.g(fsp.omega)[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_tb(fsp.omega) * rmsh.ds_b \
            # these terms constrain mu = H(omega) on the boundary
            + ((geo.H(fsp.omega) - fsp.mu) * fsp.nu_mu) * bgeo.sqrt_deth_lr(fsp.omega) * rmsh.ds_lr \
            + ((geo.H(fsp.omega) - fsp.mu) * fsp.nu_mu) * bgeo.sqrt_deth_tb(fsp.omega) * rmsh.ds_tb \
    )


# total functional for the mixed problem
F = ( F_z + F_omega + F_mu) + F_N

import variational_problem_pp_square_no_circle as vp_pp