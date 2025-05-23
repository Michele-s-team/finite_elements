import csv
import importlib
from fenics import *
import os
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

# create the path for the csv file if it does not exist
filename_bcs = rarg.args.output_directory + '/bcs.csv'
os.makedirs(os.path.dirname(filename_bcs), exist_ok=True)

csvfile = open(filename_bcs, 'a', newline='')
fieldnames = [ \
    '<<(l_profile_u_bar^i - u_bar^i)(l_profile_u_bar_i - u_bar_i)>>_l'
]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()


# this function prints out the residuals of BCs
def print_bcs():
    # get the solution and write it to file

    # write the residual of natural BCs on step 2 to file
    writer.writerows([{ \
        fieldnames[0]: \
            (sqrt(assemble((fsp.u_bar[i] - vp.u_bar_l_profile[i]) * (fsp.u_bar[i] - vp.u_bar_l_profile[i]) * rmsh.ds_l)) / \
             assemble(Constant(1.0) * rmsh.ds_l))
    }])

    csvfile.flush()
