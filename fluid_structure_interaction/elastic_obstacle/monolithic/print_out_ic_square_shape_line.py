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
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l, m, n = ufl.indices(6)

# create the path for the csv file if it does not exist
filename_ics = os.path.join(rarg.args.output_directory, 'ics.csv')
os.makedirs(os.path.dirname(filename_ics), exist_ok=True)

csvfile = open(filename_ics, 'a', newline='')
fieldnames = [ \
    '<<[v^n_i]_j [v^n_i]_j>>_{partial Omega square I}',
    '<<[varsigma]_i [varsigma]_i>>_{partial Omega square I}',
    '<<[u^n_i]_j [u^n_i]_j>>_{partial Omega circle I}',
    '<<[u^n_i]_j [u^n_i]_j>>_{partial Omega square I}',
    '<<[\dot{u}^n_i]_j [\dot{u}^n_i]_j>>_{partial Omega circle I}',
    '<<[\dot{u}^n_i]_j [\dot{u}^n_i]_j>>_{partial Omega square I}',
    '<<[v^n_i]_j [v^n_i]_j>>_{partial Omega circle}',
    '<<[varsigma]_i [varsigma]_i>>_{partial Omega circle}'
    ]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()



# this function prints out the residuals of BCs
def print_ics():

    writer.writerows([{
        fieldnames[0]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_I_square']):.{io.number_of_decimals}e}",
        fieldnames[1]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.sigma_n, bgeo.facet_normal[0])[i] * msh.jump(fsp.sigma_n, bgeo.facet_normal[0])[i]), rmsh.ds_mesh[0]['dS_I_square']):.{io.number_of_decimals}e}",
        fieldnames[2]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_I_shape']):.{io.number_of_decimals}e}",
        fieldnames[3]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_I_square']):.{io.number_of_decimals}e}",
        fieldnames[4]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_I_shape']):.{io.number_of_decimals}e}",
        fieldnames[5]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_I_square']):.{io.number_of_decimals}e}",
        fieldnames[6]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_shape']):.{io.number_of_decimals}e}",
        fieldnames[7]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.sigma_n, bgeo.facet_normal[0])[j] * msh.jump(fsp.sigma_n, bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_shape']):.{io.number_of_decimals}e}"
        }])

    csvfile.flush()
