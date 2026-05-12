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

i, j, k, l, m = ufl.indices(5)

# create the path for the csv file if it does not exist
filename_bcs = os.path.join(rarg.args.output_directory, 'bcs.csv')
os.makedirs(os.path.dirname(filename_bcs), exist_ok=True)

csvfile = open(filename_bcs, 'a', newline='')
fieldnames = [ \
    '<<|v^n - v_l|^2>>_{partial Omega l}', \
    '<<|v^n - v_tb|^2>>_{partial Omega tb}',\
    '<<|v^{n square} - average(u_dot_n)|^2>>_{partial Omega ellipse}',\
    '<<varsigma_{i 1} varsigma_{i 1}>>_{partial Omega r}'
    ]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()


# this function prints out the residuals of BCs
def print_bcs():

    writer.writerows([{
        fieldnames[0]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_l), rmsh.ds_l):.{io.number_of_decimals}e}",\
        fieldnames[1]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_tb), rmsh.ds_tb):.{io.number_of_decimals}e}",\
        fieldnames[2]: \
            f"{msh.abs_wrt_measure(sqrt((fsp.v_n(vp.sub_mesh_1_label)[i] - msh.average(fsp.u_dot_n[i])) * (fsp.v_n(vp.sub_mesh_1_label)[i] - msh.average(fsp.u_dot_n[i]))), rmsh.dS_ellipse):.{io.number_of_decimals}e}",\
        fieldnames[3]: \
            f"{msh.abs_wrt_measure(sqrt(flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_fluid'])[i, 0] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_fluid'])[i, 0]), rmsh.ds_r):.{io.number_of_decimals}e}",\
        }])

    csvfile.flush()
