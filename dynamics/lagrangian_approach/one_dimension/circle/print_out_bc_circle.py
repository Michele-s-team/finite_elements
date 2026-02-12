import colorama as col
from fenics import *
import input_output as io
import importlib
import mesh.utils as msh
import sys
import ufl as ufl

module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)


import differential_geometry.boundary.geometry as bgeo
import input_output as sys_io
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(
    f"\t\tu_[partial Omega l][0] - u_[partial Omega r][0] = {col.Fore.RED}{abs((fsp.u(rmsh.parameters['x_l'])[0] - fsp.u(rmsh.parameters['x_r'])[0])):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\tu_[partial Omega l][1] - u_[partial Omega r][1] = {col.Fore.RED}{abs((fsp.u(rmsh.parameters['x_l'])[1] - fsp.u(rmsh.parameters['x_r'])[1])):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")
