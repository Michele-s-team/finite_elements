'''
To check the solution against check-with-analytical-solution-bc-ring-2.nb
- python3 generate_ring_mesh.py 0.025
- [this point may not be needed] Make a dry run of check-with-analytical-solution-bc-ring-2.nb to generate a suitable initial condition for solve.py and uncomment the block '#uncomment this to set the initial profiles from the ODE soltion' below to read this initial condition
- clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; python3 solve.py /home/fenics/shared/steady_state/flow/mesh /home/fenics/shared/steady_state/flow/$SOLUTION_PATH
- Substitute the value of '	<z>_[partial Omega r] = ...' in 'zRmin = ...' in check-with-analytical-solution-bc-ring-2.nb so both codes have the same boundary value problem
- check-with-analytical-solution-bc-ring-2.nb
'''

'''
to produce figure-7: 

set  bc_ring_2
set everywhere
r = 0.01
R = 0.25

set

#bending rigidity
kappa = 3e-2
#density
rho = 1e-12
#viscosity
eta = 1e-2

v_r_const = -1
v_R_const = -0.040199502484483561080877059650826321556501851825
w_R_const = 0.0
sigma_R_const = 1.0
z_R_const = 0
omega_r_const = 0.1
omega_R_const = 0



'''

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


i, j, k, l, m, n, o, p = ufl.indices( 8 )


'''
CAREFUL: THIS VALUE IS NOT ARBITRARY, BUT IT MUST SATISFY THE RELATION v = C1 / ( sqrt(r * (1 + omega ** 2)))!!
If
- you fix rpam.parameters['omega_r_const'] and rpam.parameters['v_r_const'] ->
- you fix C1 ->
- if you fix rpam.parameters['omega_R_const'] -> 
- rpam.parameters['v_R_const'] is no longer arbitrary and it is given by rpam.parameters['v_R_const'] = C1 / ( sqrt(R * (1 + rpam.parameters['omega_R_const'] ** 2)))
'''




class v_r_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = rpam.parameters['v_r_const'] * x[0] / geo.my_norm( x )
        values[1] = rpam.parameters['v_r_const'] * x[1] / geo.my_norm( x )

    def value_shape(self):
        return (2,)

class v_R_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = rpam.parameters['v_R_const'] * x[0] / geo.my_norm( x )
        values[1] = rpam.parameters['v_R_const'] * x[1] / geo.my_norm( x )

    def value_shape(self):
        return (2,)




class w_R_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = rpam.parameters['w_R_const']

    def value_shape(self):
        return (1,)





class sigma_R_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = rpam.parameters['sigma_R_const']

    def value_shape(self):
        return (1,)


class z_R_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = rpam.parameters['z_R_const']

    def value_shape(self):
        return (1,)


class omega_r_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = rpam.parameters['omega_r_const'] * x[0] / geo.my_norm(x)
        values[1] = rpam.parameters['omega_r_const'] * x[1] / geo.my_norm(x)

    def value_shape(self):
        return (2,)

class omega_R_Expression( UserExpression ):
    def eval(self, values, x):

        values[0] = rpam.parameters['omega_R_const'] * x[0] / geo.my_norm(x)
        values[1] = rpam.parameters['omega_R_const'] * x[1] / geo.my_norm(x)

    def value_shape(self):
        return (2,)



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



v_r = interpolate( v_r_Expression( element=fsp.Q_v.ufl_element() ), fsp.Q_v )
v_R = interpolate( v_R_Expression( element=fsp.Q_v.ufl_element() ), fsp.Q_v )
w_R = interpolate( w_R_Expression( element=fsp.Q_w.ufl_element() ), fsp.Q_w )
sigma_R = interpolate( sigma_R_Expression( element=fsp.Q_sigma.ufl_element() ), fsp.Q_sigma )
z_R = interpolate( z_R_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
omega_r = interpolate( omega_r_Expression( element=fsp.Q_omega.ufl_element() ), fsp.Q_omega )
omega_R = interpolate( omega_R_Expression( element=fsp.Q_omega.ufl_element() ), fsp.Q_omega )


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
bc_w_R = DirichletBC( fsp.Q.sub( 1 ), w_R, rmsh.boundary_R )
bc_sigma_R = DirichletBC( fsp.Q.sub( 2 ), sigma_R, rmsh.boundary_R )
bc_z_R = DirichletBC( fsp.Q.sub( 3 ), z_R, rmsh.boundary_R )
bc_omega_r = DirichletBC( fsp.Q.sub( 4 ), omega_r, rmsh.boundary_r )
bc_omega_R = DirichletBC( fsp.Q.sub( 4 ), omega_R, rmsh.boundary_R )

# all BCs
bcs = [bc_v_r, bc_w_R, bc_sigma_R, bc_z_R, bc_omega_r, bc_omega_R]

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
            + ((bgeo.n_circle(fsp.omega)[i] * geo.g(fsp.omega)[i, j] * fsp.v[j] - bgeo.n_circle(fsp.omega)[i] * geo.g(fsp.omega)[i, j] * v_R[j]) * (bgeo.n_circle(fsp.omega)[k] * fsp.nu_v[k])) * bgeo.sqrt_deth_circle(fsp.omega, rmsh.parameters["c_R"][:2]) * rmsh.ds_R \
            # these terms constrain mu = H(omega) on the boundary
            + ((geo.H(fsp.omega) - fsp.mu) * fsp.nu_mu) * bgeo.sqrt_deth_circle(fsp.omega, rmsh.parameters["c_r"][:2]) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_r \
            + ((geo.H(fsp.omega) - fsp.mu) * fsp.nu_mu) * bgeo.sqrt_deth_circle(fsp.omega, rmsh.parameters["c_R"][:2]) * (1.0 / rmsh.parameters["R"]) * rmsh.ds_R \
    )


# total functional for the mixed problem
F = ( F_v + F_w + F_sigma + F_z + F_omega + F_mu) + F_N

import variational_problem_pp_ring as vp_pp