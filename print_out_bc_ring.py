from __future__ import print_function
from fenics import *
from mshr import *
# from variational_problem_bc_a import *
from variational_problem_bc_ring import *

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
omega_output, z_output = psi.split( deepcopy=True )


print( "\int_{\partial \Omega_r} (n^i \omega_i - psi )^2 dS = ", \
   assemble( ( (((n_circle( omega_output ))[i] * omega_output[i] - omega_r) ) ** 2 * ds_r ) )
  )
print( "\int_{\partial \Omega_R} (n^i \omega_i - psi )^2 dS = ", \
   assemble( ( (((n_circle( omega_output ))[i] * omega_output[i] - omega_R) ) ** 2 * ds_R ) )
  )