import colorama as col
from fenics import *
import importlib
import numpy as np
import termcolor
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import input_output as io
import mesh as msh

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

u_output, v_output, w_output = psi.split( deepcopy=True )

print( "BCs check: " )
print(f"<<(u - u_exact)^2>>_partial Omega = {termcolor.colored(msh.difference_on_boundary(u_output, u_exact), 'red')}")
print(f"<<(v - v_exact)^2>>_partial Omega = {termcolor.colored(msh.difference_on_boundary(v_output, v_exact), 'red')}")
print(f"<<(w - w_exact)^2>>_partial Omega = {termcolor.colored(msh.difference_on_boundary(w_output, w_exact), 'red')}")

print("Check that the PDE is satisfied: ")
print(f"<<(w - f)^2>>_Omega = {termcolor.colored(msh.difference_in_bulk(w_output, f), 'green')}")


print( "Comparison with exact solution: " )
print(f"<<(u - u_exact)^2>>_Omega = {termcolor.colored(msh.difference_in_bulk(u_output, u_exact), 'blue')}")
print(f"<<(v - v_exact)^2>>_Omega = {termcolor.colored(msh.difference_in_bulk(v_output, v_exact), 'blue')}")
print(f"<<(w - w_exact)^2>>_Omega = {termcolor.colored(msh.difference_in_bulk(w_output, w_exact), 'blue')}")



import print_out_solution