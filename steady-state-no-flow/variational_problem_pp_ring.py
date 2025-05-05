import ufl as ufl

import function_spaces as fsp
import boundary_geometry as bgeo
import geometry as geo
import read_mesh_ring as rmsh

i, j, k, l = ufl.indices( 4 )



F_pp_tau = (fsp.nu[i] * geo.g_c( fsp.omega )[i, j] * (fsp.nu_tau.dx( j )) + fsp.tau * fsp.nu_tau) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
           - ((bgeo.n_circle( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
           - ((bgeo.n_circle( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R
