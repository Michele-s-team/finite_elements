import importlib
import ufl_legacy as ufl

import function_spaces as fsp
import geometry as geo
import physics as phys
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

# post-processing variational functional
F_pp_tau = ( \
                       - vp.rho * (fsp.w_bar - fsp.w_n_1) \
                       - vp.dt * phys.conv_cn_n(fsp.v_bar, fsp.v_n_1, fsp.v_n_2, fsp.w_bar, fsp.w_n_1, fsp.omega_n_12, vp.rho) \
                       + vp.dt * phys.lhs_force_balance_equation(vp.kappa, fsp.omega_n_12, fsp.mu_n_12, fsp.sigma_n_32, fsp.tau_n_12) \
                       + vp.dt * phys.fvisc_n(fsp.V, fsp.W, fsp.omega_n_12, fsp.mu_n_12, vp.eta) \
               ) * fsp.nu_tau * geo.sqrt_detg(fsp.omega_n_12) * rmsh.dx

F_pp_d = ((geo.d(fsp.V, fsp.W, fsp.omega_n_12)[i, j] - fsp.d[i, j]) * fsp.nu_d[i, j]) * geo.sqrt_detg(fsp.omega_n_12) * rmsh.dx
