import csv
import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import elasticity as ela
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi


rmsh = importlib.import_module(swi.rmsh)
vp_mesh = importlib.import_module(swi.vp_mesh)
vp_fluid = importlib.import_module(swi.vp_fluid)

i, j, k, l, alpha, beta, gamma = ufl.indices(7)

# create the path for the csv file if it does not exist
filename_bcs = rarg.args.output_directory + '/bcs.csv'
os.makedirs(os.path.dirname(filename_bcs), exist_ok=True)

csvfile = open(filename_bcs, 'a', newline='')
fieldnames = [ \
    '<<|v_bar_fl - h|^2>>_{ds_b}',
    '<<|v_bar_fl|^2>>_{ds_l}',
    '<<|v_bar_fl_0|^2>>_{ds_r}',
    '<<|v_bar_fl - u_dot_n|^2>>_{ds_t}',
    '<<|phi_fl|^2>>_{ds_b}',
    '<<(G^{n-1}_{alpha 1 \partial V_{FL}^2 / \partial y^\alpha})^2>>_{ds_r}',
    '<<(nu_gamma G^{n-1}_{gamma alpha} G^{n-1}_{beta alpha} \\partial \\phi_{FL} / \\partial y^\\beta)^2>>_{ds_lt}',
    '<<(G^{n-1}_{beta 1} \\partial phi_{FL} / \\partial y_beta )^2>>_{ds_r}',
    
    ]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()


# this function prints out the residuals of BCs for the fluid problem
def print_bcs_fl():
    
    writer.writerows([{
        fieldnames[0]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_fl_bar - fsp.v_fl_bar_b), rmsh.ds_sub_mesh[0]['ds_b']):.{io.number_of_decimals}e}",
        fieldnames[1]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_fl_bar), rmsh.ds_sub_mesh[0]['ds_l']):.{io.number_of_decimals}e}",
        fieldnames[2]: \
            f"{msh.abs_wrt_measure(abs(fsp.v_fl_bar[0]), rmsh.ds_sub_mesh[0]['ds_r']):.{io.number_of_decimals}e}",
        fieldnames[3]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_fl_bar - fsp.u_dot_n), rmsh.ds_sub_mesh[0]['ds_t']):.{io.number_of_decimals}e}",
        fieldnames[4]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.phi_fl), rmsh.ds_sub_mesh[0]['ds_b']):.{io.number_of_decimals}e}",
        fieldnames[5]: \
            f"{msh.abs_wrt_measure(ela.G(fsp.u_n_1)[alpha, 0] * (((fsp.v_fl_n_1[1] + fsp.v_fl_bar[1]) / 2.0).dx(alpha)), rmsh.ds_sub_mesh[0]['ds_r']):.{io.number_of_decimals}e}",
        fieldnames[6]: \
            f"{msh.abs_wrt_measure(ela.G(fsp.u_n_1)[gamma, alpha] * (bgeo.sub_mesh_facet_normal[0])[gamma] * ela.G(fsp.u_n_1)[beta, alpha] * (fsp.phi_fl.dx(beta)), rmsh.ds_sub_mesh[0]['ds_l'] + rmsh.ds_sub_mesh[0]['ds_t']):.{io.number_of_decimals}e}",        
        fieldnames[7]: \
            f"{msh.abs_wrt_measure(ela.G(fsp.u_n_1)[beta, 0] * (fsp.phi_fl.dx(beta)) , rmsh.ds_sub_mesh[0]['ds_r']):.{io.number_of_decimals}e}"        
            }])

    csvfile.flush()

# print the BCs for all problems
def print_bcs():
    print_bcs_fl()