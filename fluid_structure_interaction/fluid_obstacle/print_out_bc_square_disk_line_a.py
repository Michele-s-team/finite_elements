import csv
import importlib
from fenics import *
import numpy as np
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import elasticity as ela
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
# vp_mesh = importlib.import_module(swi.vp_msh)
# vp_fluid = importlib.import_module(swi.vp_fl)

alpha = ufl.indices(1)

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

    # 4 fluid square

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
        }])

    csvfile.flush()
