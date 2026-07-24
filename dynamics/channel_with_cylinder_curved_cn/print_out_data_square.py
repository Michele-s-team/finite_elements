import csv
import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
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
    'step',
    'dTdt_L',
    'dTdt_R',
    '(dTdt_L - dTdt_R)/|(dTdt_L + dTdt_R)/2.0|'
    ]
writer = csv.DictWriter( csvfile, fieldnames=fieldnames )
writer.writeheader()


# this function prints out some useful data
def print_data(step):

    dTdt_L = assemble( ( \
                rpam.parameters['rho']/2.0 * (fsp.v_n[i] * geo.g(fsp.omega)[i, j] * fsp.v_n[j] - fsp.v_n_1[i] * geo.g(fsp.omega)[i, j] * fsp.v_n_1[j])/vp.dt \
                + fsp.v_n[i] * ( rpam.parameters['rho']/2.0 * fsp.v_n[j] * geo.g(fsp.omega)[j, k] * fsp.v_n[k] ).dx(i)
            ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx)

    dTdt_R = (assemble(  \
                            bgeo.n_lr( fsp.omega )[i] * fsp.v_n[j] * ( fsp.sigma_n_12 * geo.g(fsp.omega)[i, j] + 2 * rpam.parameters['mu'] * geo.d(fsp.v_n, fsp.w, fsp.omega )[i, j] )  * bgeo.sqrt_deth_lr( fsp.omega ) * rmsh.ds_lr \
                            + bgeo.n_tb( fsp.omega )[i] * fsp.v_n[j] * ( fsp.sigma_n_12 * geo.g(fsp.omega)[i, j] + 2 * rpam.parameters['mu'] * geo.d(fsp.v_n, fsp.w, fsp.omega )[i, j] )  * bgeo.sqrt_deth_tb( fsp.omega ) * rmsh.ds_tb \
                            + bgeo.n_circle( fsp.omega )[i] * fsp.v_n[j] * ( fsp.sigma_n_12 * geo.g(fsp.omega)[i, j] + 2 * rpam.parameters['mu'] * geo.d(fsp.v_n, fsp.w, fsp.omega )[i, j] )  * bgeo.sqrt_deth_circle( fsp.omega, rmsh.parameters["c_r"] ) * ( 1.0 / rmsh.parameters["r"]) * rmsh.ds_circle \
                        ) \
                        - assemble( ( 2.0 * rpam.parameters['mu'] * geo.d(fsp.v_n, fsp.w, fsp.omega)[j, i] * geo.d_c(fsp.v_n, fsp.w, fsp.omega)[i, j] ) * geo.sqrt_detg( fsp.omega ) * rmsh.dx))
    
    writer.writerows( [{ \
        fieldnames[0]: \
            step,\
        fieldnames[1]: \
            dTdt_L,\
        fieldnames[2]: \
            dTdt_R,\
        fieldnames[3]: \
            (dTdt_L - dTdt_R)/abs((dTdt_L + dTdt_R)/2.0),\
        }] )

    csvfile.flush()
