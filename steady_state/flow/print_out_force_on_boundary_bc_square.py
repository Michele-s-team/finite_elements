from fenics import *

import differential_geometry.boundary.geometry as bgeo
import physics.utils as phys
import parameters.read.solution as rpam
import print_out_solution as prout

# print out the force exerted on the circle
dFdl_tot_3d_to_assemble = phys.dFdl_tot_3d(prout.v_output,
                                           prout.w_output,
                                           prout.omega_output,
                                           prout.mu_output,
                                           prout.sigma_output,
                                           rpam.parameters['eta'], rpam.parameters['kappa'],
                                           bgeo.n_circle(prout.omega_output))

print("F_circle = ",\
      [assemble(dFdl * bgeo.sqrt_deth_circle(prout.omega_output, prout.rmsh.parameters["c_r"][:2]) * (1.0 / prout.rmsh.parameters["r"]) * prout.rmsh.ds_circle) for dFdl in dFdl_tot_3d_to_assemble])

