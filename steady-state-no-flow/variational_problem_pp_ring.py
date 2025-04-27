F_pp_nu = (fsp.nu[i] * fsp.nu_nu[i] + fsp.mu * geo.Nabla_v( fsp.nu_nu, fsp.omega )[i, i]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
       - ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
       - ((bgeo.n_circle( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.R) * rmsh.ds_R


F_pp_tau = (fsp.nu[i] * geo.g_c( fsp.omega )[i, j] * (fsp.nu_tau.dx( j )) + fsp.tau * fsp.nu_tau) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
           - ((bgeo.n_circle( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_r \
           - ((bgeo.n_circle( fsp.omega ))[i] * fsp.nu_tau * fsp.nu[i]) * bgeo.sqrt_deth_circle( fsp.omega, rmsh.c_R ) * (1.0 / rmsh.R) * rmsh.ds_R
