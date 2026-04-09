import importlib
import ufl as ufl

import command as cmd
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import physics.utils as phys
rmsh = importlib.import_module('mesh.read.square_no_circle')
import parameters.read.solution as rpam
import switch_problem as swi

vp = importlib.import_module(swi.vp)

cmd.set_gauge('monge')

i, j, k, l = ufl.indices( 4 )

F_pp_tau = phys.lhs_force_balance_equation(rpam.parameters["kappa"], fsp.omega, fsp.mu, fsp.sigma, fsp.tau) * fsp.nu_tau * geo.sqrt_detg(fsp.omega) * rmsh.dx
