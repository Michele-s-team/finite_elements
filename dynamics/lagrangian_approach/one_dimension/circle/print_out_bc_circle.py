import colorama as col
import csv
from fenics import *
import input_output as io
import importlib
import mesh.utils as msh
import os
import runtime_arguments as rarg
import sys
import ufl as ufl

module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)


import differential_geometry.boundary.geometry as bgeo
import input_output as sys_io
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

#set up printout of the BCs to file
# create the path for the csv file if it does not exist
os.makedirs(os.path.dirname(rarg.args.output_directory + '/bcs.csv'), exist_ok=True)

csvfile_bcs = open( (rarg.args.output_directory) + '/bcs.csv', 'a', newline='' )
fieldnames_bcs = [ \
    'u_[partial Omega l][0] - u_[partial Omega r][0]', \
    'u_[partial Omega l][1] - u_[partial Omega r][1]'
    ]
writer_bcs = csv.DictWriter( csvfile_bcs, fieldnames=fieldnames_bcs )
writer_bcs.writeheader()


def print_bcs():
     # write the residual of natural BCs on step 2 to file
    writer_bcs.writerows( [{ \
        fieldnames_bcs[0]: \
            f"{1:.{io.number_of_decimals}e}",\
        fieldnames_bcs[1]: \
            f"{2:.{io.number_of_decimals}e}"
        }] )
    csvfile_bcs.flush()


# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(
    f"\t\tu_[partial Omega l][0] - u_[partial Omega r][0] = {col.Fore.RED}{abs((fsp.u(rmsh.parameters['x_l'])[0] - fsp.u(rmsh.parameters['x_r'])[0])):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\tu_[partial Omega l][1] - u_[partial Omega r][1] = {col.Fore.RED}{abs((fsp.u(rmsh.parameters['x_l'])[1] - fsp.u(rmsh.parameters['x_r'])[1])):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")
