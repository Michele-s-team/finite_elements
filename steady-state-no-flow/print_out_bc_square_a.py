from fenics import *
import ufl as ufl
import colorama as col

import boundary_geometry as bgeo
import input_output as io
import mesh as msh
import print_out_solution as prout
import read_mesh_square as rmsh
import variational_problem_bc_square_a as vp

i, j, k, l = ufl.indices(4)



print("Check of BCs:")
print("1)")
print(
    f"\t\t<<(z - phi)^2>>_square = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, vp.z_square_const, rmsh.ds_square):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(z - phi)^2>>_circle = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, vp.z_circle_const, rmsh.ds_circle):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print("2)")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_lr = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_lr(prout.omega_output))[i] * prout.omega_output[i], vp.omega_square, rmsh.ds_lr):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_tb = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_tb(prout.omega_output))[i] * prout.omega_output[i], vp.omega_square, rmsh.ds_tb):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_circle = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_circle(prout.omega_output))[i] * prout.omega_output[i], vp.omega_circle, rmsh.ds_circle):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

'''
print( "Check if the intermediate PDEs are satisfied:" )
print(
    f"1)\t\t<<(fel + flaplace)^2>>_Omega =  {msh.difference_in_bulk( project( phys.fel_n( prout.omega_output, mu_output, fsp.tau, vp.kappa ), fsp.Q_z ), project( -phys.flaplace( fsp.sigma, prout.omega_output ), fsp.Q_z ) ):.{io.number_of_decimals}e}" )
print(
    f"2) \t\t<<|omega - partial z|^2>>_Omega = {msh.difference_in_bulk( project( sqrt( (prout.omega_output[i] - prout.z_output.dx( i )) * (prout.omega_output[i] - prout.z_output.dx( i )) ), fsp.Q_z ), project( Constant( 0 ), fsp.Q_z ) ):.{io.number_of_decimals}e}" )
print( f"3)\t\t<<[mu - H(omega)]^2>>_Omega =  {msh.difference_in_bulk( mu_output, project( geo.H( prout.omega_output ), fsp.Q_z ) ):.{io.number_of_decimals}e}" )
print( f"4)\t\t<<[tau - Nabla^i nu_i]^2>>_Omega =  {msh.difference_in_bulk( fsp.tau, project( - geo.Nabla_LB(mu_output,prout.omega_output), fsp.Q_z ) ):.{io.number_of_decimals}e}" )
'''

import print_out_forces
import print_out_force_on_circle