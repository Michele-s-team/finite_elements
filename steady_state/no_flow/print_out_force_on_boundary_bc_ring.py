from fenics import *

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import physics.utils as phys
import print_out_solution as prout

# print out the force exerted on the circle
dFdl_sigma_kappa_3d_to_assemble = phys.dFdl_sigma_kappa_3d(
    prout.omega_output,
    prout.mu_output,
    fsp.sigma,
    prout.rpam.parameters["kappa"],
    bgeo.n_circle(prout.omega_output))
dFdl_sigma_3d_to_assemble = phys.dFdl_sigma_3d(prout.omega_output, fsp.sigma, bgeo.n_circle(prout.omega_output))
dFdl_kappa_3d_to_assemble = phys.dFdl_kappa_3d(prout.omega_output, prout.mu_output, prout.rpam.parameters["kappa"], bgeo.n_circle(prout.omega_output))

print("F_{ds_r} = ", \
      [assemble(dFdl * bgeo.sqrt_deth_circle(prout.omega_output, prout.rmsh.parameters["c_r"][:2]) * (1.0 / prout.rmsh.parameters["r"]) * prout.rmsh.ds_r) for dFdl in dFdl_sigma_kappa_3d_to_assemble])

print("F_sigma_{ds_r} = ", \
      [assemble(dFdl * bgeo.sqrt_deth_circle(prout.omega_output, prout.rmsh.parameters["c_r"][:2]) * (1.0 / prout.rmsh.parameters["r"]) * prout.rmsh.ds_r) for dFdl in dFdl_sigma_3d_to_assemble])

print("F_kappa_{ds_r} = ", \
      [assemble(dFdl * bgeo.sqrt_deth_circle(prout.omega_output, prout.rmsh.parameters["c_r"][:2]) * (1.0 / prout.rmsh.parameters["r"]) * prout.rmsh.ds_r) for dFdl in dFdl_kappa_3d_to_assemble])
