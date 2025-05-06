import importlib
import ufl as ufl

import function_spaces as fsp
import geometry as geo
import physics as phys
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


i, j, k, l = ufl.indices(4)


F_pp_tau = () * fsp.nu_tau * geo.sqrt_detg(fsp.omega) * rmsh.dx

F_pp_d = ((geo.d(fsp.v, fsp.w, fsp.omega)[i, j] - fsp.d[i, j]) * fsp.nu_d[i, j]) * geo.sqrt_detg(fsp.omega) * rmsh.dx
