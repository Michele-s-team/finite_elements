from fenics import *
import importlib
import ufl as ufl

import command as cmd
import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

cmd.set_gauge('monge')

i, j, k, l = ufl.indices( 4 )

'''
To produce figure - 6 :
select bc_ring_1
set r = 0.01, R = 0.5 everywhere
refactor rpam.parameters['sigma_r_const'] -> rpam.parameters['sigma_R_const']
set 
bc_sigma_R = DirichletBC( fsp.Q.sub( 2 ), Constant( rpam.parameters['sigma_R_const'] ), rmsh.boundary_R )
print( f"\t\t<<(sigma - sigma_R)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure( sigma_output, rpam.parameters['sigma_R_const'], rmsh.ds_R ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
'''


class v_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters['v_r_const'] * x[0] / geo.my_norm(x)
        values[1] = rpam.parameters['v_r_const'] * x[1] / geo.my_norm(x)

    def value_shape(self):
        return (2,)

class z_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters['z_r_const']

    def value_shape(self):
        return (1,)


class z_R_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters['z_R_const']

    def value_shape(self):
        return (1,)

class omega_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters['omega_r_const']

    def value_shape(self):
        return (1,)


class omega_R_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters['omega_R_const']

    def value_shape(self):
        return (1,)

class v_0_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = fsp.v_0_r_read( x[0], x[1] ) * x[0] / geo.my_norm( x )
        values[1] = fsp.v_0_r_read( x[0], x[1] ) * x[1] / geo.my_norm( x )

    def value_shape(self):
        return (2,)

class w_0_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = fsp.w_0_read( x[0], x[1] )

    def value_shape(self):
        return (1,)

class sigma_0_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = fsp.sigma_0_read( x[0], x[1] )

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

class TauExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0.0
    def value_shape(self):
        return (1,)


v_r = interpolate( v_r_Expression( element=fsp.Q_v.ufl_element() ), fsp.Q_v )

z_r = interpolate( z_r_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
z_R = interpolate( z_R_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

omega_r = interpolate( omega_r_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
omega_R = interpolate( omega_R_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

#uncomment this to set the initial profiles from the ODE soltion
'''
print("Reading the initial profiles from file ...")
fu.set_from_file( fsp.v_0_r_read, 'solution-ode/v_ode.csv' )
fsp.v_0.interpolate( v_0_Expression( element=fsp.Q_v.ufl_element() ) )

fu.set_from_file( fsp.w_0_read, 'solution-ode/w_ode.csv' )
fsp.w_0.interpolate( w_0_Expression( element=fsp.Q_w.ufl_element() ) )

fu.set_from_file( fsp.sigma_0_read, 'solution-ode/sigma_ode.csv' )
fsp.sigma_0.interpolate( sigma_0_Expression( element=fsp.Q_sigma.ufl_element() ) )

fu.set_from_file( fsp.z_0_read, 'solution-ode/z_ode.csv' )
fsp.z_0.interpolate( z_0_Expression( element=fsp.Q_z.ufl_element() ) )

fu.set_from_file( fsp.omega_0_r_read, 'solution-ode/omega_ode.csv' )
fsp.omega_0.interpolate( omega_0_Expression( element=fsp.Q_omega.ufl_element() ) )

fu.set_from_file( fsp.mu_0_read, 'solution-ode/mu_ode.csv' )
fsp.mu_0.interpolate( mu_0_Expression( element=fsp.Q_mu.ufl_element() ))

# fsp.tau_0.interpolate( TauExpression( element=fsp.Q_tau.ufl_element() ) )

#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
fsp.assigner.assign(fsp.psi, [fsp.v_0, fsp.w_0, fsp.sigma_0,  fsp.z_0, fsp.omega_0, fsp.mu_0])
print("... done")
'''


# boundary conditions (BCs)
bc_v_r = DirichletBC( fsp.Q.sub( 0 ), v_r, rmsh.boundary_r )

# BCs for w_bar
bc_w_r = DirichletBC( fsp.Q.sub( 1 ), Constant( rpam.parameters['w_r_const'] ), rmsh.boundary_r )
bc_w_R = DirichletBC( fsp.Q.sub( 1 ), Constant( rpam.parameters['w_R_const'] ), rmsh.boundary_R )

#BC for sigma
bc_sigma_r = DirichletBC( fsp.Q.sub( 2 ), Constant( rpam.parameters['sigma_r_const'] ), rmsh.boundary_r )

# BCs for z
bc_z_r = DirichletBC( fsp.Q.sub( 3 ), z_r, rmsh.boundary_r )
bc_z_R = DirichletBC( fsp.Q.sub( 3 ), z_R, rmsh.boundary_R )

# all BCs
bcs = [bc_v_r, bc_w_r, bc_w_R, bc_sigma_r, bc_z_r, bc_z_R]


# Define variational problem : F_v, F_z are related to the PDEs for v, ..., z respectively . F_N enforces the BCs with Nitsche's method.
# To be safe, I explicitly wrote each term on each part of the boundary with its own normal vector and pull-back of the metric: for example, on the left (l) and on the right (r) sides of the rectangle,
# the surface elements are ds_l + ds_r, and the normal is n_lr(omega), and the pull-back of the metric is sqrt_deth_lr: this avoids odd interpolations at the corners of the rectangle edges.

F_sigma = (geo.Nabla_v( fsp.v, fsp.omega )[i, i] - 2.0 * fsp.mu * fsp.w) * fsp.nu_sigma * geo.sqrt_detg( fsp.omega ) * rmsh.dx

F_v = ( \
                    rpam.parameters['rho'] * ( \
                          (fsp.v[j] * geo.Nabla_v( fsp.v,  fsp.omega )[i, j] - 2.0 * fsp.v[j] * fsp.w * geo.g_c( fsp.omega )[i, k] * geo.b( fsp.omega )[k, j]) * fsp.nu_v[i] \
                          + 1.0 / 2.0 * (fsp.w ** 2) * geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( fsp.nu_v, fsp.omega )[i, j] \
                  ) \
                    + (fsp.sigma * geo.g_c( fsp.omega )[i, j] * geo.Nabla_f( fsp.nu_v, fsp.omega )[i, j] \
                       + 2.0 * rpam.parameters['eta'] * geo.d_c( fsp.v,  fsp.w, fsp.omega )[j, i] * geo.Nabla_f( fsp.nu_v, fsp.omega )[j, i])
      ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
      - rpam.parameters['rho'] / 2.0 * ( \
                    + ((fsp.w ** 2) * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_r"][:2] ) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_r \
                    + ((fsp.w ** 2) * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_R"][:2] ) * (1.0 / rmsh.parameters["R"]) * rmsh.ds_R
      ) \
      - ( \
                    + (fsp.sigma * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_r"][:2] ) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_r \
                    + (fsp.sigma * (bgeo.n_circle( fsp.omega ))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_R"][:2] ) * (1.0 / rmsh.parameters["R"]) * rmsh.ds_R
      ) \
      - 2.0 * rpam.parameters['eta'] * ( \
              + (geo.d_c( fsp.v,  fsp.w, fsp.omega )[i, j] * geo.g( fsp.omega )[i, k] * (bgeo.n_circle( fsp.omega ))[k] * fsp.nu_v[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_r"][:2] ) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_r \
              + (geo.d_c( fsp.v,  fsp.w, fsp.omega )[i, j] * geo.g( fsp.omega )[i, k] * (bgeo.n_circle( fsp.omega ))[k] * fsp.nu_v[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_R"][:2] ) * (1.0 / rmsh.parameters["R"]) * rmsh.ds_R
      )

F_w = ( \
                    rpam.parameters['rho'] * (fsp.v[i] * fsp.v[k] * geo.b( fsp.omega )[k, i]) * fsp.nu_w \
                    - rpam.parameters['rho'] * fsp.w * geo.Nabla_v( geo.vector_times_scalar( fsp.v,  fsp.nu_w ), fsp.omega )[i, i] \
                    + 2.0 * rpam.parameters['kappa'] * ( \
                                  - geo.g_c( fsp.omega )[i, j] * (fsp.mu.dx( i )) * (fsp.nu_w.dx( j )) \
                                  + 2.0 * fsp.mu * (fsp.mu ** 2 - geo.K( fsp.omega )) * fsp.nu_w \
                          ) \
                    - ( \
                                  2.0 * fsp.sigma * fsp.mu \
                                  + 2.0 * rpam.parameters['eta'] * (geo.g_c( fsp.omega )[i, k] * geo.Nabla_v( fsp.v,  fsp.omega )[j, k] *
                                                 (geo.b( fsp.omega ))[i, j] - 2.0 * fsp.w * (2.0 * fsp.mu ** 2 - geo.K( fsp.omega )))
                    ) * fsp.nu_w
      ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
+ rpam.parameters['rho'] * ( \
              + (fsp.w * fsp.nu_w * (bgeo.n_circle( fsp.omega ))[j] * geo.g( fsp.omega )[j, i] * fsp.v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_r"][:2] ) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_r \
              + (fsp.w * fsp.nu_w * (bgeo.n_circle( fsp.omega ))[j] * geo.g( fsp.omega )[j, i] * fsp.v[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_R"][:2] ) * (1.0 / rmsh.parameters["R"]) * rmsh.ds_R
) \
+ 2.0 * rpam.parameters['kappa'] * ( \
              + ( (bgeo.n_circle( fsp.omega ))[i] * (fsp.mu.dx( i )) * fsp.nu_w ) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_r"][:2] ) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_r \
              + ( (bgeo.n_circle( fsp.omega ))[i] * (fsp.mu.dx( i )) * fsp.nu_w ) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_R"][:2] ) * (1.0 / rmsh.parameters["R"]) * rmsh.ds_R \
)

F_z = ( \
                    - fsp.w * ((geo.normal( fsp.omega ))[2] - ((geo.normal( fsp.omega ))[0] * fsp.omega[0] + (geo.normal( fsp.omega ))[1] * fsp.omega[1])) * fsp.nu_z \
            ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx

F_omega = (fsp.z * geo.Nabla_v( fsp.nu_omega, fsp.omega )[i, i] + fsp.omega[i] * fsp.nu_omega[i]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
          - ( \
                      + ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_r"][:2] ) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_r \
                      + ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.z * fsp.nu_omega[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_R"][:2] ) * (1.0 / rmsh.parameters["R"]) * rmsh.ds_R \
              )

F_mu = ((geo.H( fsp.omega ) - fsp.mu) * fsp.nu_mu) * geo.sqrt_detg( fsp.omega ) * rmsh.dx


F_N = rpam.parameters['alpha'] / rmsh.r_mesh * ( \
            + (((bgeo.n_circle( fsp.omega ))[i] * fsp.omega[i] - omega_r) * ((bgeo.n_circle( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_r"][:2] ) * rmsh.ds_r \
            + (((bgeo.n_circle( fsp.omega ))[i] * fsp.omega[i] - omega_R) * ((bgeo.n_circle( fsp.omega ))[k] * geo.g( fsp.omega )[k, l] * fsp.nu_omega[l])) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_R"][:2] ) * rmsh.ds_R \
 \
            + ((bgeo.n_circle( fsp.omega )[i] * geo.g( fsp.omega )[i, j] * fsp.v[j] - rpam.parameters['v_R_const']) * (bgeo.n_circle( fsp.omega )[k] * fsp.nu_v[k])) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_R"][:2] ) * rmsh.ds_R \
            # these terms constrain mu = H(omega) on the boundary
            + ((geo.H(fsp.omega) - fsp.mu) * fsp.nu_mu) * bgeo.sqrt_deth_circle(fsp.omega, rmsh.parameters["c_r"][:2]) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_r \
            + ((geo.H(fsp.omega) - fsp.mu) * fsp.nu_mu) * bgeo.sqrt_deth_circle(fsp.omega, rmsh.parameters["c_R"][:2]) * (1.0 / rmsh.parameters["R"]) * rmsh.ds_R \
    )


# total functional for the mixed problem
F = ( F_v + F_w + F_sigma + F_z + F_omega + F_mu) + F_N

import variational_problem_pp_ring as vp_pp