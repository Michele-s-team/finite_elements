import csv
import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
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
    '(dTdt_L - dTdt_R)/((dTdt_L + dTdt_R)/2.0)'
    ]
writer = csv.DictWriter( csvfile, fieldnames=fieldnames )
writer.writeheader()


# this function prints out some useful data
def print_data():

    dTdt_L = assemble( ( \
                rpam.parameters['rho']/2.0 * (fsp.v_n[i] * fsp.v_n[i] - fsp.v_n_1[i] * fsp.v_n_1[i])/vp.dt \
                + fsp.v_n[i] * ( rpam.parameters['rho']/2.0 * fsp.v_n[j] * fsp.v_n[j] ).dx(i)
            ) * rmsh.dx)

    dTdt_R = (assemble( ( \
                            bgeo.facet_normal[i] * fsp.v_n[j] * flu.sigma(fsp.v_n, fsp.sigma_n_12, rpam.parameters['mu'])[i, j]
                        ) * rmsh.ds) \
                        - assemble( ( rpam.parameters['mu']/2.0 * (fsp.v_n[i].dx(j) + fsp.v_n[j].dx(i)) * (fsp.v_n[j].dx(i) + fsp.v_n[i].dx(j)) ) * rmsh.dx))
    
    writer.writerows( [{ \
        fieldnames[0]: \
            (dTdt_L - dTdt_R)/((dTdt_L + dTdt_R)/2.0)
        }] )

    csvfile.flush()
