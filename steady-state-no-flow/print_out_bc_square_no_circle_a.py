from __future__ import print_function
from fenics import *
from mshr import *
from variational_problem_bc_square_no_circle_a import *

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
omega_output, z_output = psi.split( deepcopy=True )

print( "\int_{\partial \Omega} (n^i \omega_i - psi )^2 dS = ", \
       assemble( ( ((n_lr( omega_output ))[i] * omega_output[i] - omega_l)) ** 2 * (ds_l + ds_r) ) \
       + assemble( ( ((n_tb( omega_output ))[i] * omega_output[i] - omega_l)) ** 2 * (ds_t + ds_b) ) \
       )
