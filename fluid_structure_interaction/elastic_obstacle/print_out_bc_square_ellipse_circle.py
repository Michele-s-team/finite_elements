import csv
import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import elasticity as ela
import fluid as flu
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
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
    '<<|v_bar|^2>>_{tb}', \
    '<<|v_bar - u_msh_dot_n|^2>>_ellipse', \
    '<<( mu_fluid * G_{l1} * nu_l * G_{k0} * partial_k V_i ) * ( mu_fluid * G_{m1} * nu_m * G_{n0} * partial_n V_i )>>_r', \
    '<<phi^2>>_r'
    ]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()


# natural BC for the elastic problem (force exterted by the fluid on the elastic body)
def natural_bc_el():
    return ufl.as_tensor( \
        bgeo.sub_mesh_facet_normal[0][k] * ela.N(fsp.u_el_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, k] \
        - flu.sigma_ale(fsp.v_n_1_on_sub_mesh_0, fsp.sigma_n_32_on_sub_mesh_0, fsp.u_el_n, rpam.parameters['mu_fluid'])[i, j] * geo.epsilon[j, k] * ela.F(fsp.u_el_n_1)[k, l] * fsp.dyds_ellipse[l] / sqrt(fsp.dyds_ellipse[m] * fsp.dyds_ellipse[m]), \
        (i))

# natural BC for the fluid problem (zero traction at ds_r of sub_mesh[1])
def natural_bc_fl():
    return ufl.as_tensor(
        rpam.parameters['mu_fluid'] * ela.G(fsp.u_msh_n_1)[l, 0] * bgeo.sub_mesh_facet_normal[1][l] * ela.G(fsp.u_msh_n_1)[k, 0] * (fsp.V[i].dx(k))
    , (i))


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
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_), rmsh.ds_sub_mesh[1]['ds_tb']):.{io.number_of_decimals}e}", \
        fieldnames[8]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_ - fsp.u_msh_dot_n), rmsh.ds_sub_mesh[1]['ds_ellipse']):.{io.number_of_decimals}e}", \
        fieldnames[9]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(natural_bc_fl()), rmsh.ds_sub_mesh[1]['ds_r']):.{io.number_of_decimals}e}", \
        fieldnames[10]: \
            f"{msh.abs_wrt_measure(fsp.phi, rmsh.ds_sub_mesh[1]['ds_r']):.{io.number_of_decimals}e}"
        }])

    csvfile.flush()
