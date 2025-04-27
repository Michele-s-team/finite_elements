F_pp_nu = (fsp.nu[i] * fsp.nu_nu[i] + fsp.mu * geo.Nabla_v( fsp.nu_nu, fsp.omega )[i, i]) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
       - ((bgeo.n_lr( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_lr\
       - ((bgeo.n_tb( fsp.omega ))[i] * geo.g( fsp.omega )[i, j] * fsp.mu * fsp.nu_nu[j]) * bgeo.sqrt_deth_tb( fsp.omega ) * rmsh.ds_tb

F_pp_tau = ((fsp.mu.dx(i)) * geo.g_c( fsp.omega )[i, j] * (fsp.nu_tau.dx( j )) + fsp.tau * fsp.nu_tau) * geo.sqrt_detg( fsp.omega ) * rmsh.dx \
       - ((bgeo.n_lr( fsp.omega ))[i] * fsp.nu_tau * (fsp.mu.dx(i))) * bgeo.sqrt_deth_lr( fsp.omega  ) * (rmsh.ds_l + rmsh.ds_r) \
       - ((bgeo.n_tb( fsp.omega ))[i] * fsp.nu_tau * (fsp.mu.dx(i))) * bgeo.sqrt_deth_tb( fsp.omega) * (rmsh.ds_t + rmsh.ds_b)
