import csv
import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import function_spaces as fsp
import mesh.utils as msh
import parameters.read.solution as rpam
import physics.fluid_mechanics as flu
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


i, j, k, l = ufl.indices( 4 )

# create the path for the csv file if it does not exist
filename_bcs = rarg.args.output_directory + '/bcs.csv'
os.makedirs(os.path.dirname(filename_bcs), exist_ok=True)

csvfile = open(filename_bcs, 'a', newline='' )
fieldnames = [ \
    '<<(v_n[i] - v_l[i])(v_n[i] - v_l[i])>>_{\partial Omega l}',\
    '<<(v_n[i] - v_l[i])(v_n[i] - v_l[i])>>_{\partial Omega tb circle}',\
    '<<(n_j sigma_\{ij\}) (n_k sigma_\{ik\})>>_{\partial Omega r}',\
    '<<sigma_n^2>>_{\partial Omega r}'
    ]
writer = csv.DictWriter( csvfile, fieldnames=fieldnames )
writer.writeheader()


# this function prints out the residuals of BCs
def print_bcs():
    # get the solution and write it to file

    # write the residual of natural BCs on step 2 to file
    writer.writerows( [{ \
        fieldnames[0]: \
            msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_l), rmsh.ds_l),\
        fieldnames[1]: \
            msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_tb_circle), rmsh.ds_tb + rmsh.ds_circle) ,\
        fieldnames[2]: \
            msh.abs_wrt_measure(sqrt(bgeo.facet_normal[j] * flu.sigma(fsp.v_n, fsp.sigma_n, rpam.parameters['mu'])[i, j] * bgeo.facet_normal[k] * flu.sigma(fsp.v_n, fsp.sigma_n, rpam.parameters['mu'])[i, k]), rmsh.ds_r),\
        fieldnames[3]: \
            msh.abs_wrt_measure(fsp.sigma_n, rmsh.ds_r)
        }] )

    csvfile.flush()
