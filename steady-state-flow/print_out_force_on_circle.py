from fenics import *
import ufl as ufl

import boundary_geometry as bgeo
import physics as phys
import print_out_solution as prout

# print out the force exerted on the circle
dFdl_tot_3d_to_assemble = phys.dFdl_tot_3d(prout.v_output,
                                           prout.w_output,
                                           prout.omega_output,
                                           prout.mu_output,
                                           prout.sigma_output,
                                           prout.vp.eta, prout.vp.kappa,
                                           bgeo.n_circle(prout.omega_output))
print("F_{ds_r} = ",\
      [assemble(dFdl * bgeo.sqrt_deth_circle(prout.omega_output, prout.rmsh.c_r) * (1.0 / prout.rmsh.r) * prout.rmsh.ds_r) for dFdl in dFdl_tot_3d_to_assemble])

