import importlib
import ufl as ufl

import function_spaces as fsp
import geometry as geo
import physics as phys
import read_mesh_square as rmsh
import switch_problem as swi

vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

# post-processing variational functional

# F_pp_tau = (fsp.nu_n_12[i] * geo.g_c( fsp.omega_n_12 )[i, j] * (fsp.nu_tau.dx( j )) + fsp.tau_n_12 * fsp.nu_tau) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx \
#            - ((bgeo.n_lr( fsp.omega_n_12 ))[i] * fsp.nu_tau * fsp.nu_n_12[i]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_lr \
#            - ((bgeo.n_tb( fsp.omega_n_12 ))[i] * fsp.nu_tau * fsp.nu_n_12[i]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
#            - ((bgeo.n_circle( fsp.omega_n_12 ))[i] * fsp.nu_tau * fsp.nu_n_12[i]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r


F_pp_tau = ( \
                       - vp.rho / vp.dt * (fsp.w_bar - fsp.w_n_1) \
                       - phys.conv_cn_n(fsp.v_bar, fsp.v_n_1, fsp.v_n_2, fsp.w_bar, fsp.w_n_1, fsp.omega_n_12, vp.rho) \
                       + phys.lhs_force_balance_equation(vp.kappa, fsp.omega_n_12, fsp.mu_n_12, fsp.sigma_n_32, fsp.tau_n_12) \
                       + phys.fvisc_n(fsp.V, fsp.W, fsp.omega_n_12, fsp.mu_n_12, vp.eta) \
               ) * fsp.nu_tau * geo.sqrt_detg(fsp.omega_n_12) * rmsh.dx

F_pp_d = ((geo.d(fsp.V, fsp.W, fsp.omega_n_12)[i, j] - fsp.d[i, j]) * fsp.nu_d[i, j]) * geo.sqrt_detg(fsp.omega_n_12) * rmsh.dx
