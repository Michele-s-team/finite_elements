from fenics import *
<<<<<<< Updated upstream:steady-state-no-flow/print_out_bc_ring.py
try:
    import ufl
except ImportError:
    import ufl_legacy as ufl
=======
import ufl_legacy as ufl
>>>>>>> Stashed changes:steady_state/no_flow/print_out_bc_ring.py
import colorama as col

import boundary_geometry as bgeo
import geometry as geo
import input_output as io
import mesh as msh
import print_out_solution as prout
import read_mesh_ring as rmsh
import variational_problem_bc_ring as vp

i, j, k, l = ufl.indices(4)

print("Check of BCs:")
print("1)")
print(
    f"\t\t<<(z - phi)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, vp.z_r_const, rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(z - phi)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, vp.z_R_const, rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print("2)")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_circle(prout.omega_output))[i] * prout.omega_output[i], vp.omega_r, rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_circle(prout.omega_output))[i] * prout.omega_output[i], vp.omega_R, rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print("3)")
print(
    f"\t\t<<[mu - H(omega)]^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(prout.mu_output, geo.H(prout.omega_output), rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")



import print_out_forces
import print_out_force_on_boundary_bc_ring
