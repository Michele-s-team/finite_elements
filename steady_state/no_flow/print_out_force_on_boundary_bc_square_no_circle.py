from fenics import *

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import physics.utils as phys
import print_out_solution as prout

# print out the force exerted on the square
# force per unit length exerted on the left and rigth edges of the square
dFdl_lr_sigma_kappa_3d_to_assemble = phys.dFdl_sigma_kappa_3d(
    prout.omega_output,
    prout.mu_output,
    fsp.sigma,
    prout.rpam.parameters["kappa"],
    bgeo.n_lr(prout.omega_output))


print("F_{ds_lr} = ", \
      [assemble(dFdl * bgeo.sqrt_deth_lr(prout.omega_output)  * prout.rmsh.ds_lr) for dFdl in dFdl_lr_sigma_kappa_3d_to_assemble])
