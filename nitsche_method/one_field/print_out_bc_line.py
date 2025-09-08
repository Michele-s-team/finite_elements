import colorama as col
from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)


print("Check of BCs:")
print(f"\t\t<<(u - u_constant)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, rpam.parameters['u_constant'], rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


import print_out_solution