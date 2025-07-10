import importlib
import ufl_legacy as ufl

import function_spaces as fsp
import geometry as geo
import physics as phys
import read_mesh_ring as rmsh
import switch_problem as swi

vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

F_pp_tau = phys.lhs_force_balance_equation(vp.kappa, fsp.omega, fsp.mu, fsp.sigma, fsp.tau) * fsp.nu_tau * geo.sqrt_detg(fsp.omega) * rmsh.dx
