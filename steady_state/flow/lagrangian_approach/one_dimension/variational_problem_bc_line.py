from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import command as cmd
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

cmd.set_gauge('arc_length')

i, j, k, l, alpha = ufl.indices(5)


class v_l_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters['v_l'][0]

    def value_shape(self):
        return (1,)


# uncomment this to set the initial profiles from the ODE soltion
'''
print("Reading the initial profiles from file ...")

fu.set_from_file( fsp.v_0_read, 'solution-ode/v_ode.csv' )
fsp.v_0.interpolate( v_0_Expression( element=fsp.Q_v.ufl_element() ) )

fu.set_from_file( fsp.w_0_read, 'solution-ode/w_ode.csv' )
fsp.w_0.interpolate( w_0_Expression( element=fsp.Q_w.ufl_element() ) )

fu.set_from_file( fsp.sigma_0_read, 'solution-ode/sigma_ode.csv' )
fsp.sigma_0.interpolate( sigma_0_Expression( element=fsp.Q_sigma.ufl_element() ) )

fu.set_from_file( fsp.psi_0_read, 'solution-ode/psi_ode.csv' )
fsp.psi_0.interpolate( psi_0_Expression( element=fsp.Q_psi.ufl_element() ) )

fu.set_from_file( fsp.mu_0_read, 'solution-ode/mu_ode.csv' )
fsp.mu_0.interpolate( mu_0_Expression( element=fsp.Q_mu.ufl_element() ))

fu.set_from_file( fsp.X_0_read, 'solution-ode/X_ode.csv' )
fsp.X_0.interpolate( X_0_Expression( element=fsp.Q_X.ufl_element() ))


#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., X_0
fsp.assigner.assign(fsp.psi, [fsp.v_0, fsp.w_0, fsp.sigma_0,  fsp.psi_0,  fsp.mu_0, fsp.X_0])
print("... done")
'''

v_l = interpolate(v_l_Expression(element=fsp.Q_v.ufl_element()), fsp.Q_v)

# boundary conditions (BCs)
bc_v_l = DirichletBC(fsp.Q.sub(0), v_l, rmsh.boundary_l)

bc_w = DirichletBC(fsp.Q.sub(1), Constant(rpam.parameters['w_lr']), rmsh.boundary)

bc_sigma_r = DirichletBC(fsp.Q.sub(2), Constant(rpam.parameters['sigma_r']), rmsh.boundary_r)

bc_psi_l = DirichletBC(fsp.Q.sub(3), Constant(rpam.parameters["psi_l"]), rmsh.boundary_l)
bc_psi_r = DirichletBC(fsp.Q.sub(3), Constant(rpam.parameters["psi_r"]), rmsh.boundary_r)

bc_mu_l = DirichletBC(fsp.Q.sub(4), Constant(rpam.parameters["mu_l"]), rmsh.boundary_l)

bc_X_l = DirichletBC(fsp.Q.sub(5), Constant((rpam.parameters["X_l"][0], rpam.parameters["X_l"][1])), rmsh.boundary_l)

bcs = [bc_v_l, bc_w, bc_sigma_r, bc_psi_l, bc_psi_r, bc_mu_l, bc_X_l]

# variational problem

F_sigma = (geo.Nabla_v(fsp.v, fsp.psi)[i, i] - 2.0 * fsp.mu * fsp.w) * fsp.nu_sigma * geo.sqrt_detg(fsp.psi) * rmsh.dx

F_v = ( \
                  rpam.parameters['rho'] * ( \
                      (fsp.v[j] * geo.Nabla_v(fsp.v, fsp.psi)[i, j] - 2.0 * fsp.v[j] * fsp.w * geo.g_c(fsp.psi)[i, k] * geo.b(fsp.psi)[k, j]) * fsp.nu_v[i] \
                      + 1.0 / 2.0 * (fsp.w ** 2) * geo.g_c(fsp.psi)[i, j] * geo.Nabla_f(fsp.nu_v, fsp.psi)[i, j] \
              ) \
                  + (fsp.sigma * geo.g_c(fsp.psi)[i, j] * geo.Nabla_f(fsp.nu_v, fsp.psi)[i, j] \
                     + 2.0 * rpam.parameters['eta'] * geo.d_c(fsp.v, fsp.w, fsp.psi)[j, i] * geo.Nabla_f(fsp.nu_v, fsp.psi)[j, i])
      ) * geo.sqrt_detg(fsp.psi) * rmsh.dx \
      - rpam.parameters['rho'] / 2.0 * ( \
                  ((fsp.w ** 2) * (bgeo.n_lr(fsp.psi))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
          ) \
      - ( \
                  (fsp.sigma * (bgeo.n_lr(fsp.psi))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
          ) \
      - 2.0 * rpam.parameters['eta'] * ( \
                  + (geo.d_c(fsp.v, fsp.w, fsp.psi)[i, j] * geo.g(fsp.psi)[i, k] * (bgeo.n_lr(fsp.psi))[k] * fsp.nu_v[j]) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds_l \
          )

F_psi = ( \
                    rpam.parameters['rho'] * (fsp.v[i] * fsp.v[k] * geo.b(fsp.psi)[k, i]) * fsp.nu_psi \
                    - rpam.parameters['rho'] * fsp.w * geo.Nabla_v(geo.vector_times_scalar(fsp.v, fsp.nu_psi), fsp.psi)[i, i] \
                    + 2.0 * rpam.parameters['kappa'] * ( \
                                - geo.g_c(fsp.psi)[i, j] * (fsp.mu.dx(i)) * (fsp.nu_psi.dx(j)) \
                                + 2.0 * fsp.mu * (fsp.mu ** 2 - geo.K(fsp.psi)) * fsp.nu_psi \
                        ) \
                    - ( \
                                2.0 * fsp.sigma * fsp.mu \
                                + 2.0 * rpam.parameters['eta'] * (geo.g_c(fsp.psi)[i, k] * geo.Nabla_v(fsp.v, fsp.psi)[j, k] *
                                                                  (geo.b(fsp.psi))[i, j] - 2.0 * fsp.w * (2.0 * fsp.mu ** 2 - geo.K(fsp.psi)))
                    ) * fsp.nu_psi
        ) * geo.sqrt_detg(fsp.psi) * rmsh.dx \
        + rpam.parameters['rho'] * ( \
                    + (fsp.w * fsp.nu_psi * (bgeo.n_lr(fsp.psi))[j] * geo.g(fsp.psi)[j, i] * fsp.v[i]) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
            ) \
        + 2.0 * rpam.parameters['kappa'] * ( \
                    + ((bgeo.n_lr(fsp.psi))[i] * (fsp.mu.dx(i)) * fsp.nu_psi) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
            )

F_mu = ((fsp.mu - geo.H(fsp.psi)) * fsp.nu_mu) * geo.sqrt_detg(fsp.psi) * rmsh.dx

F_w = (fsp.w * fsp.nu_w) * geo.sqrt_detg(fsp.psi) * rmsh.dx

F_X = (fsp.X[alpha].dx(0) - geo.e(fsp.psi)[0, alpha]) * fsp.nu_X[alpha] * rmsh.dx

F_N = rpam.parameters["alpha"] / rmsh.r_mesh * ( \
    # these terms constrain mu = H(psi) on the boundary
        ((fsp.mu - geo.H(fsp.psi)) * fsp.nu_mu) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
    )

# total functional for the mixed problem
F = (F_v + F_w + F_sigma + F_psi + F_mu + F_X) + F_N
# sign
