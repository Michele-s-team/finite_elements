import csv
import importlib
from fenics import *
import numpy as np
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import elasticity as ela
import fluid as flu
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp_fluid_di = importlib.import_module(swi.vp_fluid_di)

alpha, beta, gamma, delta = ufl.indices(4)

# residual of natural BC [Eq. (115)] for the disk fluid problem 
def natural_bc_fl_di_v__():
    return ufl.as_tensor(
        (
             flu.sigma_ale(fsp.V_di, fsp.sigma_disk_n_32, fsp.u_n_1_di, rpam.parameters['eta_di'])[alpha, beta] * ela.G(fsp.u_n_1_di)[gamma, beta] * bgeo.sub_mesh_facet_normal[0][0][gamma]
        ) \
        - (
             flu.sigma_ale(fsp.v_square_n_1_0_1_on_0_0, fsp.sigma_square_n_32_0_1_on_0_0, fsp.u_n_1_di, rpam.parameters['eta_sq'])[alpha, beta] * ela.G(fsp.u_n_1_di)[gamma, beta] * bgeo.sub_mesh_facet_normal[0][0][gamma] + 1.0 / ela.detF(fsp.u_n_1_di) * vp_fluid_di.f_M(fsp.c_n_1, fsp.U_n_32)[alpha] 
        ), 
        (alpha)
    )

# residual of natural BC [Eq. (118)] for the square fluid problem 
def natural_bc_fl_sq():
    return ufl.as_tensor(
        ela.detF(fsp.u_n_1_sq) * flu.sigma_ale(fsp.V_sq, fsp.sigma_square_n_32, fsp.u_n_1_sq, rpam.parameters['eta_sq'])[alpha, beta] * ela.G(fsp.u_n_1_sq)[delta, beta] * (- bgeo.sub_mesh_facet_normal[0][1][delta]) + fsp.t_sq_n[alpha], 
        (alpha)
    )

# create the path for the csv file if it does not exist
filename_bcs = rarg.args.output_directory + '/bcs.csv'
os.makedirs(os.path.dirname(filename_bcs), exist_ok=True)

csvfile = open(filename_bcs, 'a', newline='')
fieldnames = [ \

    # 1 bcs for I
    '|U_n_12_[partial Omega - l] - U_n_12_[partial Omega - r]|', \
    
    # 2 bcs for D
    # 2.1 disk
    '<<|u_n_di - U_n_12|^2>>_[partial Omega^y circle]', \
    '<<|u_n_di_dot - [v_square^{n-1} \dot \hat{n}^{n-1/2}] \hat{n}^{n-1/2}|^2>>_[partial Omega^y circle]', \
    # 2.2 square
    '<<|u_n_sq|^2>>_[partial Omega^y sq]', \
    '<<|u_n_sq - U_n_12|^2>>_[partial Omega^y circle]',\
    '<<|u_n_sq_dot|^2>>_[partial Omega^y sq]', \
    '<<|u_n_sq_dot - [v_square^{n-1} \dot \hat{n}^{n-1/2}] \hat{n}^{n-1/2}|^2>>_[partial Omega^y circle]',\
    
    # 3 fluid disk
    '<<|\varsigma_{\alpha \beta}^disk G^{n-1}_{\gamma \beta} \nu_gamma - [\varsigma_{\alpha \beta}^square G^{n-1}_{\gamma \beta} \nu_\gamma + 1/|F^{n-1}| F_M^\alpha]|^2>>_[partial Omega^y circle]', \
    '<<BC_F_N^2>>_[partial Omega^y circle]', \

    # 4 fluid square
    '<<|v_square__ - g^n|^2>>_[(partial Omega^y square in) U (partial Omega^y square out) U (partial Omega^y square b)]', \
    '<<|v_square__ - v^n_circle|^2>>_[partial Omega^y circle]', \
    '<<||F^{n-1}| \\varsigma^square_{\alpha \beta} G^{n-1}_{\delta \beta} \nu_\delta + \textrm{t}^n_\alpha|^2>>_[partial Omega^y t]', \
    '<<|phi_square|^2>>_[partial Omega^y square t]', \
    '<<(G^{n-1}_{\gamma \alpha} \partial \phi_square / \partial y_\gamma G^{n-1}_{\beta \alpha} \nu_\beta)^2>>_[(partial Omega^y square in) U (partial Omega^y square out) U (partial Omega^y square b) U (partial Omega^y circle)]', \
        
    # 5 M

    ]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

# this function prints out the residuals of BCs
def print_bcs():
    # write the residual of natural BCs  to file
    writer.writerows([{
        
        #1 I

        fieldnames[0]: \
            f"{np.sqrt((fsp.U_n_12(lmsh.mesh_parameters[1]['x_l'])[0] - fsp.U_n_12(lmsh.mesh_parameters[1]['x_r'])[0])**2 + (fsp.U_n_12(lmsh.mesh_parameters[1]['x_l'])[1] - fsp.U_n_12(lmsh.mesh_parameters[1]['x_r'])[1])**2):.{io.number_of_decimals}e}",\
            

        # 2 D

        # 2.1 disk
        fieldnames[1]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n_di - fsp.U_n_12_1_on_0_0), rmsh.ds_sub_mesh[0][0]['ds']):.{io.number_of_decimals}e}",\
        fieldnames[2]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n_di_dot - fsp.u_n_di_dot_bc_di), rmsh.ds_sub_mesh[0][0]['ds']):.{io.number_of_decimals}e}",\
            
        # 2.2 square
        fieldnames[3]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n_sq), rmsh.ds_sub_mesh[0][1]['ds_lrtb']):.{io.number_of_decimals}e}",\
        fieldnames[4]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n_sq - fsp.U_n_12_1_on_0_1), rmsh.ds_sub_mesh[0][1]['ds_circle']):.{io.number_of_decimals}e}",\
        fieldnames[5]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n_sq_dot), rmsh.ds_sub_mesh[0][1]['ds_lrtb']):.{io.number_of_decimals}e}",\
        fieldnames[6]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n_sq_dot - fsp.u_n_sq_dot_bc_di), rmsh.ds_sub_mesh[0][1]['ds_circle']):.{io.number_of_decimals}e}",\


        #3 fluid disk
        fieldnames[7]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(natural_bc_fl_di_v__()), rmsh.ds_sub_mesh[0][0]['ds']):.{io.number_of_decimals}e}",\
        fieldnames[8]: \
            f"{msh.abs_wrt_measure(vp_fluid_di.natural_bc_fl_di_phi(), rmsh.ds_sub_mesh[0][0]['ds']):.{io.number_of_decimals}e}",\
            
        # 4 fluid square
        fieldnames[9]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_square__ - fsp.v_square__bc), rmsh.ds_sub_mesh[0][1]['ds_lr'] + rmsh.ds_sub_mesh[0][1]['ds_b']):.{io.number_of_decimals}e}",\
        fieldnames[10]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_square__ - fsp.v_disk_n_0_0_on_0_1), rmsh.ds_sub_mesh[0][1]['ds_circle']):.{io.number_of_decimals}e}",\
        fieldnames[11]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(natural_bc_fl_sq()), rmsh.ds_sub_mesh[0][1]['ds_t']):.{io.number_of_decimals}e}",\
        fieldnames[12]: \
            f"{msh.abs_wrt_measure(fsp.phi_square, rmsh.ds_sub_mesh[0][1]['ds_t']):.{io.number_of_decimals}e}",\
        fieldnames[13]: \
            f"{msh.abs_wrt_measure(ela.G(fsp.u_n_1_sq)[gamma, alpha] * (fsp.phi_square.dx(gamma)) * ela.G(fsp.u_n_1_sq)[beta, alpha] * (- bgeo.sub_mesh_facet_normal[0][1][beta]) , rmsh.ds_sub_mesh[0][1]['ds_lr'] + rmsh.ds_sub_mesh[0][1]['ds_b'] + rmsh.ds_sub_mesh[0][1]['ds_circle']):.{io.number_of_decimals}e}",\
        }])

    csvfile.flush()
