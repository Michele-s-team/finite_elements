import ufl as ufl

import function_spaces as fsp
import boundary_geometry as bgeo
import geometry as geo
import read_mesh_square as rmsh

i, j, k, l = ufl.indices( 4 )

#post-processing variational functional
F_pp_nu = (fsp.nu_n_12[i] * fsp.nu_nu[i] + fsp.mu_n_12 * geo.Nabla_v( fsp.nu_nu, fsp.omega_n_12 )[i, i]) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx \
       - ((bgeo.n_lr( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.mu_n_12 * fsp.nu_nu[j]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_lr \
       - ((bgeo.n_tb( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.mu_n_12 * fsp.nu_nu[j]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
       - ((bgeo.n_circle( fsp.omega_n_12 ))[i] * geo.g( fsp.omega_n_12 )[i, j] * fsp.mu_n_12 * fsp.nu_nu[j]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r

F_pp_tau = (fsp.nu_n_12[i] * geo.g_c( fsp.omega_n_12 )[i, j] * (fsp.nu_tau.dx( j )) + fsp.tau_n_12 * fsp.nu_tau) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx \
           - ((bgeo.n_lr( fsp.omega_n_12 ))[i] * fsp.nu_tau * fsp.nu_n_12[i]) * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_lr \
           - ((bgeo.n_tb( fsp.omega_n_12 ))[i] * fsp.nu_tau * fsp.nu_n_12[i]) * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
           - ((bgeo.n_circle( fsp.omega_n_12 ))[i] * fsp.nu_tau * fsp.nu_n_12[i]) * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r

F_pp_d = ((geo.d(fsp.V, fsp.W, fsp.omega_n_12)[i, j] - fsp.d[i, j]) * fsp.nu_d[i, j]) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx
