import importlib
import ufl as ufl

import command as cmd
import differential_geometry.manifold.geometry as geo
import function_spaces as fsp
import physics.utils as phys
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

cmd.set_gauge('monge')

i, j, k, l = ufl.indices(4)

bcs_tau = []
bcs_d = []

# tau is determined by solving Eq. (3c) in 'Notes'
F_pp_tau = ( \
                       - phys.conv_cn_n(fsp.v, fsp.v, fsp.v, fsp.w, fsp.w, fsp.omega, rpam.parameters['rho']) \
                       + phys.lhs_force_balance_equation(rpam.parameters['kappa'], fsp.omega, fsp.mu, fsp.sigma, fsp.tau) \
                       + phys.fvisc_n(fsp.v, fsp.w, fsp.omega, fsp.mu, rpam.parameters['eta']) \
               ) * fsp.nu_tau * geo.sqrt_detg(fsp.omega) * rmsh.dx

F_pp_d = ((geo.d(fsp.v, fsp.w, fsp.omega)[i, j] - fsp.d[i, j]) * fsp.nu_d[i, j]) * geo.sqrt_detg(fsp.omega) * rmsh.dx
