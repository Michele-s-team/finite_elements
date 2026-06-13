import importlib
import ufl as ufl


import command as cmd
import differential_geometry.manifold.geometry as geo
import function_spaces as fsp
import parameters.read.solution as rpam
import physics.utils as phys
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

cmd.set_gauge('monge')


i, j, k, l = ufl.indices(4)

bcs_tau = []
bcs_d = []

# post-processing variational functional
F_pp_tau = ( \
                       - rpam.parameters['rho'] * (fsp.w_bar - fsp.w_n_1) \
                       - vp.dt * phys.conv_cn_n(fsp.v_bar, fsp.v_n_1, fsp.v_n_2, fsp.w_bar, fsp.w_n_1, fsp.omega_n_12, rpam.parameters['rho']) \
                       + vp.dt * phys.lhs_force_balance_equation(rpam.parameters['kappa'], fsp.omega_n_12, fsp.mu_n_12, fsp.sigma_n_32, fsp.tau_n_12) \
                       + vp.dt * phys.fvisc_n(fsp.V, fsp.W, fsp.omega_n_12, fsp.mu_n_12, rpam.parameters['eta']) \
               ) * fsp.nu_tau * geo.sqrt_detg(fsp.omega_n_12) * rmsh.dx

F_pp_d = ((geo.d(fsp.V, fsp.W, fsp.omega_n_12)[i, j] - fsp.d[i, j]) * fsp.nu_d[i, j]) * geo.sqrt_detg(fsp.omega_n_12) * rmsh.dx
