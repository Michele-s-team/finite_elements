import colorama as col
from fenics import *
import importlib
import numpy as np
import termcolor
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import input_output as io
import mesh.mesh as msh

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

z_output, omega_output, mu_output, rho_output, tau_output = fsp.psi.split( deepcopy=True )

print( "Check of BCs: " )
print( f"\t<<(z - z_exact)^2>>_partial Omega = {termcolor.colored( msh.difference_on_boundary( z_output, fsp.z_exact ), 'red' )}" )
print(
    f"\t<<|omega - omega_exact|^2>>_partial Omega = {termcolor.colored( np.sqrt( assemble( (bgeo.facet_normal[i] * omega_output[i] - bgeo.facet_normal[i] * fsp.omega_exact[i]) ** 2 * rmsh.ds ) / assemble( Constant( 1 ) * rmsh.ds ) ), 'red' )}" )
print( f"\t<<(mu - mu_exact)^2>>_partial Omega = {termcolor.colored( msh.difference_on_boundary( mu_output, fsp.mu_exact ), 'red' )}" )
print(
    f"\t<<|rho - rho_exact|^2>>_partial Omega = {termcolor.colored( np.sqrt( assemble( (rho_output[i] - fsp.rho_exact[i]) * (rho_output[i] - fsp.rho_exact[i]) * rmsh.ds ) / assemble( Constant( 1 ) * rmsh.ds ) ), 'red' )}" )
print( f"\t<<(tau - tau_exact)^2>>_partial Omega = {termcolor.colored( msh.difference_on_boundary( tau_output, fsp.f ), 'red' )}" )

'''
print( "Check that the PDE is satisfied: " )
print( f"\t<<(Nabla^2 partial_i ( z partial_i z) - f)^2>>_Omega = {termcolor.colored( msh.difference_in_bulk( tau_output, tau_exact ), 'green' )}" )

print( "Comparison with exact solution: " )
print( f"\t<<(z - z_exact)^2>>_Omega = {termcolor.colored( msh.difference_in_bulk( z_output, z_exact ), 'blue' )}" )
print(
    f"\t<<|omega - omega_exact|^2>>_Omega = {termcolor.colored( msh.difference_in_bulk( project( sqrt( (omega_output[i] - omega_exact[i]) * (omega_output[i] - omega_exact[i]) ), Q_z ), project( Constant( 0 ), Q_z ) ), 'blue' )}" )
print( f"\t<<(mu - mu_exact)^2>>_Omega = {termcolor.colored( msh.difference_in_bulk( mu_output, mu_exact ), 'blue' )}" )
print(
    f"\t<<|rho - rho_exact|^2>>_Omega = {termcolor.colored( msh.difference_in_bulk( project( sqrt( (rho_output[i] - rho_exact[i]) * (rho_output[i] - rho_exact[i]) ), Q_z ), project( Constant( 0 ), Q_z ) ), 'blue' )}" )
print( f"\t<<(tau - tau_exact)^2>>_Omega = {termcolor.colored( msh.difference_in_bulk( tau_output, tau_exact ), 'blue' )}" )
'''

import print_out_solution