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

z_output, u_output, omega_output, mu_output = fsp.psi.split(deepcopy=True)

print("Check of BCs: ")
print(f"\t<<(z - z_exact)^2>>_partial Omega = {termcolor.colored(msh.difference_on_boundary(z_output, fsp.z_exact), 'red')}")
print(f"\t<<(mu - mu_exact)^2>>_partial Omega = {termcolor.colored(msh.difference_on_boundary(mu_output, fsp.mu_exact), 'red')}")

import print_out_solution
