import colorama as col
from fenics import *
import ufl as ufl

import boundary_geometry as bgeo
import geometry as geo
import input_output as io
import mesh as msh
import physics as phys
import read_mesh_square as rmsh

import variational_problem_bc_square_a as vp

i, j, k, l = ufl.indices(4)

import print_out_solution as prout

print("Check of BCs:")
print(
    f"\t\t<<|v^i - v_l^i|^2>>_[partial Omega l] = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((prout.v_output[i] - vp.v_l[i]) * (prout.v_output[i] - vp.v_l[i])), rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(v^i n_i)^2>>_[partial Omega tb] = {col.Fore.RED}{msh.abs_wrt_measure(bgeo.n_tb(prout.omega_output)[i] * geo.g(prout.omega_output)[i, j] * prout.v_output[j], rmsh.ds_tb):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(v^i n_i)^2>>_[partial Omega circle] = {col.Fore.RED}{msh.abs_wrt_measure(bgeo.n_circle(prout.omega_output)[i] * geo.g(prout.omega_output)[i, j] * prout.v_output[j], rmsh.ds_circle):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<(w - w_boundary)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(prout.w_output, vp.w_boundary, rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<(sigma - sigma_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout.sigma_output, vp.sigma_r, rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<(z - z_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, vp.z_circle, rmsh.ds_circle):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(z - z_square)^2>>_[partial Omega square] = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, vp.z_square, rmsh.ds_square):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<(n^i \omega_i - omega_r )^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_circle(prout.omega_output))[i] * prout.omega_output[i], vp.omega_circle, rmsh.ds_circle):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(n^i \omega_i - omega_square )^2>>_[partial Omega lr] = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_lr(prout.omega_output))[i] * prout.omega_output[i], vp.omega_square, rmsh.ds_lr):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(n^i \omega_i - omega_square )^2>>_[partial Omega tb] = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_tb(prout.omega_output))[i] * prout.omega_output[i], vp.omega_square, rmsh.ds_tb):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

# print out the force exerted on the circle
print(
    f"F_circle = {[assemble(phys.dFdl(prout.v_output, prout.w_output, prout.omega_output, prout.sigma_output, vp.eta, geo.n_c_r(bgeo.mesh, rmsh.c_r, prout.omega_output))[0] * bgeo.sqrt_deth_circle(prout.omega_output, rmsh.c_r) * (1.0 / rmsh.r) * rmsh.ds_circle), assemble(phys.dFdl(prout.v_output, prout.w_output, prout.omega_output, prout.sigma_output, vp.eta, geo.n_c_r(bgeo.mesh, rmsh.c_r, prout.omega_output))[1] * bgeo.sqrt_deth_circle(prout.omega_output, rmsh.c_r) * (1.0 / rmsh.r) * rmsh.ds_circle)]}")
