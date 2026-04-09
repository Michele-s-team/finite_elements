from fenics import *
import importlib

import differential_geometry.boundary.geometry as bgeo
import physics.utils as phys
import print_out_solution as prout
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


# print out the force exerted on the circle
dFdl_tot_3d_to_assemble = phys.dFdl_tot_3d(prout.v_output,
                                           prout.w_output,
                                           prout.omega_output,
                                           prout.mu_output,
                                           prout.sigma_output,
                                           rpam.parameters['eta'], rpam.parameters['kappa'],
                                           bgeo.n_circle(prout.omega_output))

print("F_{ds_r} = ",\
      [assemble(dFdl * bgeo.sqrt_deth_circle(prout.omega_output, rmsh.parameters["c_r"][:2]) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_r) for dFdl in dFdl_tot_3d_to_assemble])

