from fenics import *
import importlib
import ufl as ufl


import command as cmd
import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import function as fu
import function_spaces as fsp
import input_output as io
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

cmd.set_gauge('arc_length')


i, j, k, l, alpha = ufl.indices( 5 )


dt = rpam.parameters['T'] / rpam.parameters['N']


# uncomment this to set the initial profiles from the ODE soltion
'''
print("Reading the initial profiles from file ...")
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'v_bar.csv', fsp.v_bar_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'w_bar.csv', fsp.w_bar_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'phi.csv', fsp.phi_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'v_n.csv', fsp.v_n_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'w_n.csv', fsp.w_n_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'X_n_12.csv', fsp.X_n_12_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'nu_n_12.csv', fsp.nu_n_12_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'psi_n_12.csv', fsp.psi_n_12_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'mu_n_12.csv', fsp.mu_n_12_0)

fsp.assigner.assign(fsp.phi, [fsp.v_bar_0, fsp.w_bar_0, fsp.phi_0, fsp.v_n_0, fsp.w_n_0, fsp.X_n_12_0, fsp.nu_n_12_0, fsp.psi_n_12_0, fsp.mu_n_12_0 ])
print('... done')
'''


# boundary conditions
bc_v_bar_l = DirichletBC(fsp.Q.sub(0), Constant((rpam.parameters['v_bar_l'])), rmsh.boundary_l)
bc_v_bar_r = DirichletBC(fsp.Q.sub(0), Constant((rpam.parameters['v_bar_r'])), rmsh.boundary_r)

bc_w_bar = DirichletBC(fsp.Q.sub(1), Constant(rpam.parameters['w_bar_lr']), rmsh.boundary)

bc_phi_r = DirichletBC(fsp.Q.sub(2), Constant(0), rmsh.boundary_r)

bc_X_n_12_l = DirichletBC(fsp.Q.sub(5), Constant((rpam.parameters["X_n_12_l"][0], rpam.parameters["X_n_12_l"][1])), rmsh.boundary_l)
bc_X_n_12_r = DirichletBC(fsp.Q.sub(5), Constant((rpam.parameters["X_n_12_r"][0], rpam.parameters["X_n_12_r"][1])), rmsh.boundary_r)


# all BCs
bcs = [bc_v_bar_l, bc_v_bar_r, bc_w_bar, bc_phi_r, bc_X_n_12_l, bc_X_n_12_r]
'''
\b(kappa|rho|eta|T|N|alpha|quadrature_degree|function_space_degree)\b
'''
# sign

# Define variational problem : F_vbar, F_wbar .... F_mu_n_12 are related to the PDEs for v_bar, ..., mu^{n-1/2} respectively .  

F_v_bar = ( \
                      rpam.parameters['rho'] * (( \
                                         (fsp.v_bar[i] - fsp.v_n_1[i]) \
                                         + dt * ((3.0 / 2.0 * fsp.v_n_1[j] - 1.0 / 2.0 * fsp.v_n_2[j]) * geo.Nabla_v( fsp.V, fsp.psi_n_12, fsp.nu_n_12 )[i, j] \
                                                     - 2.0 * fsp.V[j] * fsp.W * geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, k] * geo.b( fsp.psi_n_12, fsp.nu_n_12 )[k, j]) \
 \
                                 ) * fsp.nu_v_bar[i] \
                             + dt * 1.0 / 2.0 * (fsp.W ** 2) * geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, j] * geo.Nabla_f( fsp.nu_v_bar, fsp.psi_n_12, fsp.nu_n_12 )[i, j] \
                             ) \
                      + dt * (fsp.sigma_n_32 * geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, j] * geo.Nabla_f( fsp.nu_v_bar, fsp.psi_n_12, fsp.nu_n_12 )[i, j] \
                                  + 2.0 * rpam.parameters['eta'] * geo.d_c( fsp.V, fsp.W, fsp.psi_n_12, fsp.nu_n_12 )[i, j] * geo.Nabla_f( fsp.nu_v_bar, fsp.psi_n_12, fsp.nu_n_12 )[j, i])
          ) * geo.sqrt_detg( fsp.psi_n_12, fsp.nu_n_12 ) * rmsh.dx \
          - dt * rpam.parameters['rho'] / 2.0 * ( \
                      ((fsp.W ** 2) * (bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds \
          ) \
          - dt * ( \
                      (fsp.sigma_n_32 * (bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12 ))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds \
           ) \
          - dt * 2.0 * rpam.parameters['eta'] * ( \
                      (geo.d_c( fsp.V, fsp.W, fsp.psi_n_12, fsp.nu_n_12 )[i, j] * geo.g( fsp.psi_n_12, fsp.nu_n_12 )[i, k] * (bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12 ))[k] * fsp.nu_v_bar[j]) * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds
          )


F_w_bar = ( \
                      rpam.parameters['rho'] * ((fsp.w_bar - fsp.w_n_1) + dt * fsp.V[i] * fsp.V[k] * geo.b( fsp.psi_n_12, fsp.nu_n_12 )[k, i]) * fsp.nu_w_bar \
                      - dt * rpam.parameters['rho'] * fsp.W * geo.Nabla_v( geo.vector_times_scalar( 3.0 / 2.0 * fsp.v_n_1 - 1.0 / 2.0 * fsp.v_n_2, fsp.nu_w_bar ), fsp.psi_n_12, fsp.nu_n_12 )[i, i] \
                      + dt * 2.0 * rpam.parameters['kappa'] * ( \
                                  - geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, j] * ((fsp.mu_n_12).dx( j )) * (fsp.nu_w_bar.dx( i )) \
                                  + 2.0 * fsp.mu_n_12 * (((fsp.mu_n_12) ** 2) - geo.K( fsp.psi_n_12, fsp.nu_n_12 )) * fsp.nu_w_bar \
                          ) \
                      - dt * ( \
                                  2.0 * fsp.sigma_n_32 * fsp.mu_n_12 \
                                  + 2.0 * rpam.parameters['eta'] * (geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, k] * geo.Nabla_v( fsp.V, fsp.psi_n_12, fsp.nu_n_12 )[j, k] *
                                                 (geo.b( fsp.psi_n_12, fsp.nu_n_12 ))[i, j] - 2.0 * fsp.W * (
                                                         2.0 * ((fsp.mu_n_12) ** 2) - geo.K( fsp.psi_n_12, fsp.nu_n_12 )))
                      ) * fsp.nu_w_bar
          ) * geo.sqrt_detg( fsp.psi_n_12, fsp.nu_n_12 ) * rmsh.dx \
          + dt * rpam.parameters['rho'] * ( \
                      (fsp.W * fsp.nu_w_bar * (bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12 ))[j] * geo.g( fsp.psi_n_12, fsp.nu_n_12 )[j, i] * (3.0 / 2.0 * fsp.v_n_1[i] - 1.0 / 2.0 * fsp.v_n_2[i])) * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds \

          ) \
          + dt * 2.0 * rpam.parameters['kappa'] * ( \
                      (fsp.nu_w_bar * (bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12 ))[i] * ((fsp.mu_n_12).dx( i ))) * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds \
          )

# sign


F_phi = ( \
                    dt * geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, j] * (fsp.phi.dx( i )) * (fsp.nu_phi.dx( j )) \
                    + rpam.parameters['rho'] * (geo.Nabla_v( fsp.v_bar, fsp.psi_n_12, fsp.nu_n_12 )[i, i] - 2.0 * fsp.mu_n_12 * fsp.w_bar) * fsp.nu_phi \
            ) * geo.sqrt_detg( fsp.psi_n_12, fsp.nu_n_12 ) * rmsh.dx \
    # natural BC implemented here
- dt * ((bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12 ))[i] * (fsp.phi.dx( i )) * fsp.nu_phi) * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds_r

'''

F_v_n = ((rpam.parameters['rho'] * (fsp.v_n[i] - fsp.v_bar[i]) + dt * geo.g_c( fsp.omega_n_12 )[i, j] * (fsp.phi.dx( j ))) * fsp.nu_v_n[i]) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx

F_w_n = ((fsp.w_n - fsp.w_bar) * fsp.nu_w_n) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx

F_z_n = ( \
                    ( \
                                (fsp.z_n_12 - fsp.z_n_32) \
                                - dt * fsp.w_n_1 * ((geo.normal( fsp.omega_n_12 ))[2] - ((geo.normal( fsp.omega_n_12 ))[0] * fsp.omega_n_12[0] + (geo.normal( fsp.omega_n_12 ))[1] * fsp.omega_n_12[1])) \
                        ) * fsp.nu_z_n_12 \
            ) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx

F_omega_n = (fsp.z_n_12 * geo.Nabla_v( fsp.nu_omega_n_12, fsp.omega_n_12 )[i, i] + fsp.omega_n_12[i] * fsp.nu_omega_n_12[i]) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx \
            - ( \
                        ((bgeo.n_lr( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.z_n_12 * fsp.nu_omega_n_12[j]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * (rmsh.ds_l + rmsh.ds_r) \
                        + ((bgeo.n_tb( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.z_n_12 * fsp.nu_omega_n_12[j]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * (rmsh.ds_t + rmsh.ds_b) \
                        + ((bgeo.n_circle( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.z_n_12 * fsp.nu_omega_n_12[j]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.parameters["c_r"] ) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_circle
            )

F_mu_n = ((geo.H( fsp.omega_n_12 ) - fsp.mu_n_12) * fsp.nu_mu_n_12) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx


F_N = rpam.parameters['alpha'] / rmsh.r_mesh * ( \
 \
            (fsp.v_bar[i] * geo.g( fsp.omega_n_12 )[i, j] * (bgeo.n_tb( fsp.omega_n_12 ))[j] * (bgeo.n_tb( fsp.omega_n_12 ))[k] * fsp.nu_v_bar[k]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * (rmsh.ds_t + rmsh.ds_b) \
            + (fsp.v_bar[i] * geo.g( fsp.omega_n_12 )[i, j] * (bgeo.n_circle( fsp.omega_n_12 ))[j] * (bgeo.n_circle( fsp.omega_n_12 ))[k] * fsp.nu_v_bar[k]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.parameters["c_r"] ) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_circle \
 \
            + (((bgeo.n_lr( fsp.omega_n_12 ))[i] * fsp.omega_n_12[i] - grad_square) * (bgeo.n_lr( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, k] * fsp.nu_omega_n_12[k]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * (rmsh.ds_l + rmsh.ds_r) \
            + (((bgeo.n_tb( fsp.omega_n_12 ))[i] * fsp.omega_n_12[i] - grad_square) * (bgeo.n_tb( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, k] * fsp.nu_omega_n_12[k]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * (rmsh.ds_t + rmsh.ds_b) \
            + (((bgeo.n_circle( fsp.omega_n_12 ))[i] * fsp.omega_n_12[i] - grad_circle) * (bgeo.n_circle( fsp.omega_n_12 ))[j] * geo.g( fsp.omega_n_12 )[j, k] * fsp.nu_omega_n_12[k]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.parameters["c_r"] ) * (
                    1.0 / rmsh.parameters["r"]) * rmsh.ds_circle \
            # these terms constrain mu_n_12 = H(omega_n_12) on the boundary
            + ((geo.H(fsp.omega_n_12) - fsp.mu_n_12) * fsp.nu_mu_n_12) * bgeo.sqrt_deth_lr(fsp.omega_n_12) * rmsh.ds_lr \
            + ((geo.H(fsp.omega_n_12) - fsp.mu_n_12) * fsp.nu_mu_n_12) * bgeo.sqrt_deth_tb(fsp.omega_n_12) * rmsh.ds_tb \
            + ((geo.H(fsp.omega_n_12) - fsp.mu_n_12) * fsp.nu_mu_n_12) * bgeo.sqrt_deth_circle(fsp.omega_n_12, rmsh.parameters["c_r"]) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_circle \
    )

# total functional for the mixed problem
F = (F_v_bar + F_w_bar + F_phi + F_v_n + F_w_n + F_z_n + F_omega_n + F_mu_n) + F_N

'''