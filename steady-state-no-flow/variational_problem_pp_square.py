import ufl as ufl

import function_spaces as fsp
import boundary_geometry as bgeo
import geometry as geo
import read_mesh_square as rmsh

i, j, k, l = ufl.indices( 4 )

F_pp_nu = (fsp.nu[i] * fsp.nu_nu[i] + fsp.mu * geo.Nabla_v(fsp.nu_nu, fsp.omega)[i, i]) * geo.sqrt_detg(fsp.omega) * rmsh.dx \
          - ((bgeo.n_lr(fsp.omega))[i] * geo.g(fsp.omega)[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_lr(fsp.omega) * rmsh.ds_lr \
          - ((bgeo.n_tb(fsp.omega))[i] * geo.g(fsp.omega)[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_tb(fsp.omega) * rmsh.ds_tb \
          - ((bgeo.n_circle(fsp.omega))[i] * geo.g(fsp.omega)[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_circle(fsp.omega, rmsh.c_r) * (1.0 / rmsh.r) * rmsh.ds_circle

F_pp_tau = ((fsp.mu.dx(i)) * geo.g_c(fsp.omega)[i, j] * (fsp.nu_tau.dx(j)) + fsp.tau * fsp.nu_tau) * geo.sqrt_detg(fsp.omega) * rmsh.dx \
           - ((bgeo.n_lr(fsp.omega))[i] * fsp.nu_tau * (fsp.mu.dx(i))) * bgeo.sqrt_deth_lr(fsp.omega) * rmsh.ds_lr \
           - ((bgeo.n_tb(fsp.omega))[i] * fsp.nu_tau * (fsp.mu.dx(i))) * bgeo.sqrt_deth_tb(fsp.omega) * rmsh.ds_tb \
           - ((bgeo.n_circle(fsp.omega))[i] * fsp.nu_tau * (fsp.mu.dx(i))) * bgeo.sqrt_deth_circle(fsp.omega, rmsh.c_r) * (1.0 / rmsh.r) * rmsh.ds_circle
