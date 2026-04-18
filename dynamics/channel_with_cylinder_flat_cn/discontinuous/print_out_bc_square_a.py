import csv
import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import physics.fluid_mechanics as flu
import function_spaces as fsp
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


alpha, beta, gamma = ufl.indices( 3 )

# create the path for the csv file if it does not exist
filename_bcs = rarg.args.output_directory + '/bcs.csv'
os.makedirs(os.path.dirname(filename_bcs), exist_ok=True)

csvfile = open(filename_bcs, 'a', newline='' )
fieldnames = [ \
    '<<|sigma_{alpha beta} n_beta - tau_alpha|^2>>_{partial Omega r}',\
    '<<phi^2>>_{partial Omega r}'
    ]
writer = csv.DictWriter( csvfile, fieldnames=fieldnames )
writer.writeheader()


# this function prints out the residuals of BCs
def print_bcs():
    # get the solution and write it to file


    # write the residual of natural BCs on step 2 to file
    writer.writerows( [{ \
        fieldnames[0]: \
            msh.abs_wrt_measure(sqrt((flu.sigma(fsp.V, fsp.sigma_n_32, rpam.parameters['mu'])[alpha, beta] * bgeo.facet_normal[beta] - fsp.tau[alpha]) * (flu.sigma(fsp.V, fsp.sigma_n_32, rpam.parameters['mu'])[alpha, gamma] * bgeo.facet_normal[gamma] - fsp.tau[alpha])), rmsh.ds_r),\
        fieldnames[1]:
            msh.abs_wrt_measure(fsp.phi, rmsh.ds_r)
        }] )

    csvfile.flush()
