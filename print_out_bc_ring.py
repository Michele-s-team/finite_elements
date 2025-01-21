from __future__ import print_function
from fenics import *
from mshr import *
# from variational_problem_bc_a import *
from variational_problem_bc_ring import *

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
z_output, omega_output, mu_output, nu_output = psi.split( deepcopy=True )

print("Check of BCs:")
print( "\t<<(n^i \omega_i - psi )^2>>_r = ", \
   sqrt(assemble( ( (((n_circle( omega_output ))[i] * omega_output[i] - omega_r) ) ** 2 * ds_r ) ) / assemble(Constant(1.0) * ds_r))
  )
print( "\t<<(n^i \omega_i - psi )^2>>_R = ", \
   sqrt(assemble( ( (((n_circle( omega_output ))[i] * omega_output[i] - omega_R) ) ** 2 * ds_R ) ) / assemble( Constant(1.0) * ds_R))
  )