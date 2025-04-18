import colorama as col
from fenics import *
import ufl as ufl

import boundary_geometry as bgeo
import geometry as geo
import input_output as io
import mesh as msh
import read_mesh_ring as rmsh

import variational_problem_bc_ring_2 as vp

i, j, k, l = ufl.indices(4)

import print_out_solution as prout

print("Check of BCs:")
print(
    f"\t\t<<|v^i - v_r^i|^2>>_[partial Omega r] = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((prout.v_output[i] - vp.v_r[i]) * (prout.v_output[i] - vp.v_r[i])), rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(n^i v_i - n^i v_R_i)|^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure(bgeo.n_circle(prout.omega_output)[i] * geo.g(prout.omega_output)[i, j] * prout.v_output[j], bgeo.n_circle(prout.omega_output)[i] * geo.g(prout.omega_output)[i, j] * vp.v_R[j], rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
# print( f"\t\t<<(n_i n_j Pi^[ij])^2>>_[partial Omega R] = {col.Fore.RED}{msh.abs_wrt_measure((bgeo.n_circle( prout.omega_output )[i] * geo.g( prout.omega_output )[i, j] * bgeo.n_circle( prout.omega_output )[k] * geo.g( prout.omega_output )[k, l] * phys.Pi( prout.v_output, w_output, prout.omega_output, prout.sigma_output, vp.eta )[j, l]), rmsh.ds_R ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )

print(
    f"\t\t<<(w - w_R)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure(prout.w_output, vp.w_R, rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<(sigma - sigma_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout.sigma_output, vp.sigma_R, rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<(z - phi)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, vp.z_R, rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<|\omega_i - omega_r_i |^2>>_[partial Omega r] = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((prout.omega_output[i] - vp.omega_r[i]) * (prout.omega_output[i] - vp.omega_r[i])), rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<|\omega_i - omega_R_i |^2>>_[partial Omega R] = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((prout.omega_output[i] - vp.omega_R[i]) * (prout.omega_output[i] - vp.omega_R[i])), rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\n\t\t<z>_[partial Omega r] = {col.Fore.YELLOW}{assemble(prout.z_output * rmsh.ds_r) / assemble(Constant(1.0) * rmsh.ds_r)}{col.Style.RESET_ALL}")
