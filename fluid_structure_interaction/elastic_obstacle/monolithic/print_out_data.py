'''
this module prints some useful data (mean displacement of the elastic body, pressure at the interface ... ) to monitor the time iteration
'''

import csv
import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import physics.fluid_mechanics as flu
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j = ufl.indices(2)

# create the path for the csv file if it does not exist
filename_data = os.path.join(rarg.args.output_directory, 'data.csv')
os.makedirs(os.path.dirname(filename_data), exist_ok=True)

csvfile = open(filename_data, 'a', newline='')
fieldnames = [ \
    '<<|u_n|^2>>_{partial Omega ellipse}',
    '<<sigma_n^2>>_{partial Omega ellipse}',
    ]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()



# this function prints out the residuals of BCs
def print_data():

    writer.writerows([{
        fieldnames[0]: \
            f"{sqrt(assemble(msh.average(fsp.u_n[i]*fsp.u_n[i]) * rmsh.dS_ellipse)):.{io.number_of_decimals}e}",
        fieldnames[1]: \
            f"{sqrt(assemble((fsp.sigma_n(vp.sub_mesh_1_label))**2 * rmsh.dS_ellipse)):.{io.number_of_decimals}e}"
        }])

    csvfile.flush()
