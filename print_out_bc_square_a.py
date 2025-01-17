from __future__ import print_function
from fenics import *
from mshr import *
from physics import *
from variational_problem_bc_square_a import *

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
v_output, w_output, sigma_output, omega_output, z_output = psi.split( deepcopy=True )

print( "\int_{\partial \Omega_W U \partial \Omega_O} (n_i v^i)^2 dS = ", \
       assemble( ((n_tb(omega))[i] * g( omega_output )[i, j] * v_output[j]) ** 2 * (ds_t + ds_b) ) \
       + assemble( ((n_circle(omega))[i] * g( omega_output )[i, j] * v_output[j]) ** 2 * ds_circle ) \
       )

print( "\int_{\partial \Omega} (n^i \omega_i - psi )^2 dS = ", \
       assemble( ( ((n_lr( omega_output ))[i] * omega_output[i] - omega_square)) ** 2 * (ds_l + ds_r) ) \
       + assemble( ( ((n_tb( omega_output ))[i] * omega_output[i] - omega_square)) ** 2 * (ds_t + ds_b) ) \
       + assemble( ( ((n_circle( omega_output ))[i] * omega_output[i] - omega_circle)) ** 2 * ds_circle ) \
       )

print( "\int_{\partial \Omega OUT} ( n_i d^{i 1})^2 dS = ", \
       assemble( (d_c( v_output, w_output, omega_output )[i, 0] * g( omega_output )[i, j] * (n_lr( omega_output ))[j]) ** 2 * ds_r ) \
    )

print("Forces:")
print(f"\tF_l = [{assemble( dFdl(v_output, w_output, omega_output, sigma_output, eta, n_lr(omega_output))[0] * sqrt_deth_lr( omega ) * ds_l )}, {assemble( dFdl(v_output, w_output, omega_output, sigma_output, eta, n_lr(omega_output))[1] * sqrt_deth_lr( omega ) * ds_l )}]")
print(f"\tF_r = [{assemble( dFdl(v_output, w_output, omega_output, sigma_output, eta, n_lr(omega_output))[0] * sqrt_deth_lr( omega ) * ds_r )}, {assemble( dFdl(v_output, w_output, omega_output, sigma_output, eta, n_lr(omega_output))[1] * sqrt_deth_lr( omega ) * ds_r )}]")
