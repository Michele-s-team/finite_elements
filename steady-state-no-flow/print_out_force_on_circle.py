from fenics import *

import boundary_geometry as bgeo
import function_spaces as fsp
import physics as phys
import print_out_solution as prout

# print out the force exerted on the circle
dFdl_sigma_kappa_3d_to_assemble = phys.dFdl_sigma_kappa_3d(
    prout.omega_output,
    prout.mu_output,
    fsp.sigma,
    prout.vp.kappa,
    bgeo.n_circle(prout.omega_output))
dFdl_sigma_3d_to_assemble = phys.dFdl_sigma_3d(prout.omega_output, fsp.sigma, bgeo.n_circle(prout.omega_output))
dFdl_kappa_3d_to_assemble = phys.dFdl_kappa_3d(prout.omega_output, prout.mu_output, prout.vp.kappa, bgeo.n_circle(prout.omega_output))

print("F_circle = ", \
      [assemble(dFdl * bgeo.sqrt_deth_circle(prout.omega_output, prout.rmsh.c_r) * (1.0 / prout.rmsh.r) * prout.rmsh.ds_circle) for dFdl in dFdl_sigma_kappa_3d_to_assemble])

print("{F_sigma}_circle = ", \
      [assemble(dFdl * bgeo.sqrt_deth_circle(prout.omega_output, prout.rmsh.c_r) * (1.0 / prout.rmsh.r) * prout.rmsh.ds_circle) for dFdl in dFdl_sigma_3d_to_assemble])

print("F_kappa_circle = ", \
      [assemble(dFdl * bgeo.sqrt_deth_circle(prout.omega_output, prout.rmsh.c_r) * (1.0 / prout.rmsh.r) * prout.rmsh.ds_circle) for dFdl in dFdl_kappa_3d_to_assemble])
