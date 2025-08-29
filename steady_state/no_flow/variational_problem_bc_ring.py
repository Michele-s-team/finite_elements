from fenics import *
import importlib
import numpy as np
import ufl as ufl

import command as cmd
import function_spaces as fsp
import differential_geometry.boundary.geometry as bgeo
import differential_geometry.boundary.gauges.monge_gauge as bgeo_g
import differential_geometry.manifold.geometry as geo
import differential_geometry.manifold.gauges.monge_gauge as geo_g
import read_parameters_solve as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

# in geo, set the gauge-specific methods equal to the methods for the arc-length gauge
cmd.set_global_variables(geo, \
                         { \
                             'e': geo_g.e, \
                             'K': geo_g.K, \
                             'normal': geo_g.normal,\
                             'X': geo_g.X} \
                         )

# in bgeo, set the gauge-specific methods equal to the methods for the arc-length gauge
cmd.set_global_variables(bgeo, \
                         { \
                             'Nt_circle': bgeo_g.Nt_circle, \
                             'Nn_circle': bgeo_g.Nn_circle, \
                             'dydtheta': bgeo_g.dydtheta, \
                             'sqrt_deth_circle': bgeo_g.sqrt_deth_circle, \
                             'sqrt_deth_lr': bgeo_g.sqrt_deth_lr, \
                             'sqrt_deth_tb': bgeo_g.sqrt_deth_tb, \
                             'Nt_lr': bgeo_g.Nt_lr, \
                             'Nn_lr': bgeo_g.Nn_lr, \
                             'Nt_tb': bgeo_g.Nt_tb, \
                             'Nn_tb': bgeo_g.Nn_tb, \
                             'n_lr': bgeo_g.n_lr, \
                             'n_tb': bgeo_g.n_tb, \
                             'n_circle': bgeo_g.n_circle\
                             } \
                         )

i, j, k, l = ufl.indices( 4 )


# values of z at the boundaries
'''
if you compare with the solution from check-with-analytical-solution-bc-ring.nb:
    - z_r(R)_const_{here} <-> zRmin(max)_{check-with-analytical-solution-bc-ring.nb}
    - zp_r(R)_const_{here} <-> zpRmin(max)_{check-with-analytical-solution-bc-ring.nb}
'''

omega_r_const = - (rmsh.parameters["r"]) * rpam.parameters["zp_r_const"] / np.sqrt( (rmsh.parameters["r"]) ** 2 * (1.0 + rpam.parameters["zp_r_const"] ** 2) )
omega_R_const = (rmsh.parameters["R"]) * rpam.parameters["zp_R_const"] / np.sqrt( (rmsh.parameters["R"]) ** 2 * (1.0 + rpam.parameters["zp_R_const"] ** 2) )



class SurfaceTensionExpression( UserExpression ):
    def eval(self, values, x):
        values[0] =  rpam.parameters["sigma_const"]
        # values[0] = ((2 + rpam.parameters["C"]**2) * rpam.parameters["kappa"]) / (2 * (1 + rpam.parameters["C"]**2) * geo.my_norm(x)**2)

    def value_shape(self):
        return (1,)


class z_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters["C"] * geo.my_norm( x )

    def value_shape(self):
        return (1,)


class omega_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters["C"] * x[0] / (geo.my_norm( x ))
        values[1] = rpam.parameters["C"] * x[1] / (geo.my_norm( x ))

    def value_shape(self):
        return (2,)


class mu_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters["C"] / (2.0 * np.sqrt( 1.0 + rpam.parameters["C"] ** 2 ) * geo.my_norm( x ))

    def value_shape(self):
        return (1,)


class tau_exact_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters["C"] / (2.0 * ((1.0 + rpam.parameters["C"] ** 2) * (geo.my_norm( x )) ** 2) ** (3.0 / 2.0))

    def value_shape(self):
        return (1,)


class z_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters["z_r_const"]

    def value_shape(self):
        return (1,)


class z_R_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters["z_R_const"]

    def value_shape(self):
        return (1,)


class omega_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = omega_r_const

    def value_shape(self):
        return (1,)


class omega_R_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = omega_R_const

    def value_shape(self):
        return (1,)


class z_0_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = fsp.z_0_read( x[0], x[1] )

    def value_shape(self):
        return (1,)

class omega_0_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = fsp.omega_0_r_read(x[0], x[1]) * x[0] / geo.my_norm(x)
        values[1] = fsp.omega_0_r_read(x[0], x[1]) * x[1] / geo.my_norm(x)

    def value_shape(self):
        return (2,)


class mu_0_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = fsp.mu_0_read( x[0], x[1] )

    def value_shape(self):
        return (1,)



# values of z on ds_r and ds_R, to be used to check if the boundary conditions (BCs) are satisfied
z_r = interpolate( z_r_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
z_R = interpolate( z_R_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

# values of \partial_i z = omega_i on the ds_r and ds_R, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
omega_r = interpolate( omega_r_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
omega_R = interpolate( omega_R_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

fsp.sigma.interpolate( SurfaceTensionExpression( element=fsp.Q_sigma.ufl_element() ) )

fsp.z_exact.interpolate( z_exact_Expression( element=fsp.Q_z.ufl_element() ) )
fsp.omega_exact.interpolate( omega_exact_Expression( element=fsp.Q_omega.ufl_element() ) )
fsp.mu_exact.interpolate( mu_exact_Expression( element=fsp.Q_mu.ufl_element() ) )


#uncomment this to set the initial profiles from the ODE soltion
'''
print("Reading the initial profiles from file ...")
fu.set_from_file( fsp.z_0_read, 'solution-ode/z_ode.csv' )
fsp.z_0.interpolate( z_0_Expression( element=fsp.Q_z.ufl_element() ) )

fu.set_from_file( fsp.omega_0_r_read, 'solution-ode/omega_ode.csv' )
fsp.omega_0.interpolate( omega_0_Expression( element=fsp.Q_omega.ufl_element() ) )

fu.set_from_file( fsp.mu_0_read, 'solution-ode/mu_ode.csv' )
fsp.mu_0.interpolate( mu_0_Expression( element=fsp.Q_mu.ufl_element() ))

fsp.tau_exact.interpolate( tau_exact_Expression( element=fsp.Q_tau.ufl_element() ) )

#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
fsp.assigner.assign(fsp.psi, [fsp.z_0, fsp.omega_0, fsp.mu_0])
print("... done")
'''

fsp.tau_0.interpolate( tau_exact_Expression( element=fsp.Q_tau.ufl_element() ) )


# boundary conditions (BCs)

# bc_z = DirichletBC( fsp.Q.sub( 0 ), fsp.z_exact, rmsh.boundary )
bc_z_r = DirichletBC( fsp.Q.sub( 0 ), rpam.parameters["z_r_const"], rmsh.boundary_r )
bc_z_R = DirichletBC( fsp.Q.sub( 0 ), rpam.parameters["z_R_const"], rmsh.boundary_R )

# all BCs
bcs = [bc_z_r, bc_z_R]
# bcs = [bc_z, bc_tau]

# Define variational problem

F_z = (rpam.parameters["kappa"] * (geo.g_c( fsp.omega )[i, j] * (fsp.mu.dx(j)) * (fsp.nu_z.dx( i )) - 2.0 * fsp.mu * ((fsp.mu ** 2) - geo.K( fsp.omega )) * fsp.nu_z) + fsp.sigma * fsp.mu * fsp.nu_z) * geo.sqrt_detg(
    fsp.omega ) * rmsh.dx \
      - ( \
                  + (rpam.parameters["kappa"] * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_z * (fsp.mu.dx(i))) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_r"][:2] ) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_r \
                  + (rpam.parameters["kappa"] * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_z * (fsp.mu.dx(i))) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_R"][:2] ) * (1.0 / rmsh.parameters["R"]) * rmsh.ds_R
      )

F_omega = (- fsp.z * geo.Nabla_v( fsp.nu_omega, fsp.omega )[i, i] - fsp.omega[i] * fsp.nu_omega[i]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
          + ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_r"][:2] ) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_r \
          + ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_R"][:2] ) * (1.0 / rmsh.parameters["R"]) * rmsh.ds_R

F_mu = ((geo.H( fsp.omega ) - fsp.mu) * fsp.nu_mu) * geo.sqrt_detg( fsp.omega ) * rmsh.dx

F_N = rpam.parameters["alpha"] / rmsh.r_mesh * ( \
            + (((bgeo.n_circle( fsp.omega ))[i] * fsp.omega[i] - omega_r) * ((bgeo.n_circle( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_circle( fsp.omega,
                                                                                                                                                                                     rmsh.parameters["c_r"][:2] ) * (
                    1.0 / rmsh.parameters["r"]) * rmsh.ds_r \
            + (((bgeo.n_circle( fsp.omega ))[i] * fsp.omega[i] - omega_R) * ((bgeo.n_circle( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_circle( fsp.omega,
                                                                                                                                                                                     rmsh.parameters["c_R"][:2] ) * (
                    1.0 / rmsh.parameters["R"]) * rmsh.ds_R \
            # these terms constrain mu = H(omega) on the boundary
            + ((geo.H(fsp.omega) - fsp.mu) * fsp.nu_mu) * bgeo.sqrt_deth_circle(fsp.omega, rmsh.parameters["c_r"][:2]) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_r \
            + ((geo.H(fsp.omega) - fsp.mu) * fsp.nu_mu) * bgeo.sqrt_deth_circle(fsp.omega, rmsh.parameters["c_R"][:2]) * (1.0 / rmsh.parameters["R"]) * rmsh.ds_R \
    )

# total functional for the mixed problem
F = (F_z + F_omega + F_mu ) + F_N

import variational_problem_pp_ring as vp_pp

