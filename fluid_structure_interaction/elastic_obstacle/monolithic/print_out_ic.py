'''
this module prints the ICs (internal conditions) relative to the interior facets of the mesh
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

i, j, k, l, m, n = ufl.indices(6)

# create the path for the csv file if it does not exist
filename_ics = os.path.join(rarg.args.output_directory, 'ics.csv')
os.makedirs(os.path.dirname(filename_ics), exist_ok=True)

csvfile = open(filename_ics, 'a', newline='')
fieldnames = [ \
    '<<[v^n_i]_j [v^n_i]_j>>_{partial Omega square I}',
    ]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()



# this function prints out the residuals of BCs
def print_ics():

    writer.writerows([{
        fieldnames[0]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.v_n[i], bgeo.facet_normal)[j] * msh.jump(fsp.v_n[i], bgeo.facet_normal)[j]), rmsh.dS_I[1]):.{io.number_of_decimals}e}",
        }])

    csvfile.flush()
