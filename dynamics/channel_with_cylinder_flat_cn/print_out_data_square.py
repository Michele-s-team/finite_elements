import csv
import importlib
from fenics import *
import os
import ufl as ufl

import function_spaces as fsp
import parameters.read.solution as rpam
import physics.fluid_mechanics as flu
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


i, j, k, l = ufl.indices( 4 )

# create the path for the csv file if it does not exist
filename_data = rarg.args.output_directory + '/data.csv'
os.makedirs(os.path.dirname(filename_data), exist_ok=True)

csvfile = open(filename_data, 'a', newline='' )
fieldnames = [ \
    'dE/dt',\
    ]
writer = csv.DictWriter( csvfile, fieldnames=fieldnames )
writer.writeheader()


# this function prints out some useful data
def print_data():
    
    writer.writerows( [{ \
        fieldnames[0]: \
            assemble( ( \
                rpam.parameters['rho']/2.0 * (fsp.v_n[i] * fsp.v_n[i] - fsp.v_n_1[i] * fsp.v_n_1[i])/vp.dt \
                + fsp.v_n[i] * ( rpam.parameters['rho']/2.0 * fsp.v_n[j] * fsp.v_n[j] ).dx(i)
            ) * rmsh.dx), \
        }] )

    csvfile.flush()
