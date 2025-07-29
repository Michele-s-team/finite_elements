import csv
import importlib
from fenics import *
import os
import ufl as ufl

import boundary_geometry as bgeo
import elasticity as ela
import function_spaces as fsp
import geometry as geo
import input_output as io
import mesh as msh
import read_parameters_solve as rpam
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp_mesh = importlib.import_module(swi.vp_msh)
vp_fluid = importlib.import_module(swi.vp_fl)

i, j, k, l, m = ufl.indices(5)

# create the path for the csv file if it does not exist
filename_bcs = rarg.args.output_directory + '/bcs.csv'
os.makedirs(os.path.dirname(filename_bcs), exist_ok=True)

csvfile = open(filename_bcs, 'a', newline='')
fieldnames = [ \
    # bcs for elastic problem
    '<<|u_el^n|^2>>_circle', \
    '<<(nu_j P_{ij} - varsigma_{ij} epsilon_{jk} F_{kl} dy[l]/ds / |dy/ds}) * (nu_m P_{im} - varsigma_{im} epsilon_{mn} F_{no} dy[o]/ds / |dy/ds})>>_square',\
    # bcs for mesh problem
    '<<|u_msh_n - u_el_n|^2>>_ellipse', \
    '<<|u_msh_n|^2>>_square', \
    '<<|u_msh_dot_n - u_el_dot_n|^2>>_ellipse', \
    '<<|u_msh_dot_n|^2>>_square', \
    # bcs for fluid problem
    '<<|l_profile_v_bar - v_bar|^2>>_l' , \
    '<<|v_bar|^2>>_{tb}'
    # , \
    # '<<v_bar^i v_bar_i>>_{tb}', \
    # '<<(ellipse_profile_v_bar^i - v_bar^i)(v__profile_ellipse - v_bar_i)>>_ellipse', \
    # '<<\mu_fluid G^{n-1}_{j1} \partial_j V_i>>_r', \
    # '<<(G^{n-1}_{ji} nu_j G^{n-1}_{li} \partial_l phi)^2>>_{l + tb + ellipse}' ,\
    # '<<phi^2>>_r'
]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()


def natural_bc_el():
    return ufl.as_tensor( \
        bgeo.sub_mesh_facet_normal[0][k] * ela.P(fsp.u_el_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, k] \
        - ela.var_sigma_tensor(fsp.sigma_n_32_on_sub_mesh_0, fsp.v_n_1_on_sub_mesh_0, fsp.u_el_n, rpam.parameters['mu_fluid'])[i, j] * geo.epsilon[j, k] * ela.F(fsp.u_el_n_1)[k, l] * fsp.dyds_ellipse[l] / sqrt(fsp.dyds_ellipse[m] * fsp.dyds_ellipse[m]), \
        (i))




# this function prints out the residuals of BCs
def print_bcs():
    # write the residual of natural BCs  to file
    writer.writerows([{
        # bcs for elastic problem
        fieldnames[0]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_el_n), rmsh.ds_sub_mesh[0]['ds_circle']):.{io.number_of_decimals}e}"
        , \
        fieldnames[1]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(natural_bc_el()), rmsh.ds_sub_mesh[0]['ds_ellipse']):.{io.number_of_decimals}e}", \
        # bcs for mesh problem
        fieldnames[2]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_msh_n - fsp.u_el_n_on_sub_mesh_1), rmsh.ds_sub_mesh[1]['ds_ellipse']):.{io.number_of_decimals}e}", \
        fieldnames[3]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_msh_n), rmsh.ds_sub_mesh[1]['ds_lrtb']):.{io.number_of_decimals}e}", \
        fieldnames[4]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_msh_dot_n - fsp.u_el_dot_n_on_sub_mesh_1), rmsh.ds_sub_mesh[1]['ds_ellipse']):.{io.number_of_decimals}e}", \
        fieldnames[5]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_msh_dot_n), rmsh.ds_sub_mesh[1]['ds_lrtb']):.{io.number_of_decimals}e}", \
        # bcs for fluid problem
        fieldnames[6]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(vp_fluid.v__profile_l - fsp.v_), rmsh.ds_sub_mesh[1]['ds_l']):.{io.number_of_decimals}e}", \
        fieldnames[7]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_), rmsh.ds_sub_mesh[1]['ds_tb']):.{io.number_of_decimals}e}"
           }])

    csvfile.flush()
