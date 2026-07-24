import csv
import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
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
    'T',\
    ]
writer = csv.DictWriter( csvfile, fieldnames=fieldnames )
writer.writeheader()


# this function prints out some useful data
def print_data():
    
    writer.writerows( [{ \
        fieldnames[0]: \
            (0), \
        }] )

    csvfile.flush()
