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

i, j, k, l = ufl.indices(4)

# create the path for the csv file if it does not exist
filename_bcs = rarg.args.output_directory + '/bcs.csv'
os.makedirs(os.path.dirname(filename_bcs), exist_ok=True)

csvfile = open(filename_bcs, 'a', newline='')
fieldnames = [ \
    '<<(u^n_i - u_polygon_i)(u^n_i - u_polygon_i)>>_polygon', \
    '<<(u^n_i - u_square_i)(u^n_i - u_square_i)>>_square', \
    '<<(u_dot^n_i - u_dot_polygon_i)(u_dot^n_i - u_dot_polygon_i)>>_polygon', \
    '<<(u_dot^n_i - u_dot_square_i)(u_dot^n_i - u_dot_square_i)>>_square', \
    '<<(l_profile_v_bar^i - v_bar^i)(l_profile_v_bar_i - v_bar_i)>>_l', \
    '<<v_bar^i v_bar_i>>_{tb}', \
    '<<(polygon_profile_v_bar^i - v_bar^i)(v__profile_polygon - v_bar_i)>>_polygon', \
    '<<\mu G^{n-1}_{j1} \partial_j V_i>>_r', \
    '<<(G^{n-1}_{ji} nu_j G^{n-1}_{li} \partial_l phi)^2>>_{l + tb + polygon}' ,\
    '<<phi^2>>_r'
    ]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()


# this function prints out the residuals of BCs
def print_bcs():
    # write the residual of natural BCs  to file
    writer.writerows([{
        fieldnames[0]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_polygon - fsp.u_n), rmsh.ds_poly):.{io.number_of_decimals}e}", \
        fieldnames[1]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_square - fsp.u_n), rmsh.ds_square):.{io.number_of_decimals}e}", \
        fieldnames[2]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_dot_polygon - fsp.u_dot_n), rmsh.ds_poly):.{io.number_of_decimals}e}", \
        fieldnames[3]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_dot_square - fsp.u_dot_n), rmsh.ds_square):.{io.number_of_decimals}e}", \
        fieldnames[4]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(vp_fluid.v__profile_l - fsp.v_), rmsh.ds_l):.{io.number_of_decimals}e}", \
        fieldnames[5]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_), rmsh.ds_tb):.{io.number_of_decimals}e}", \
        fieldnames[6]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(vp_fluid.v__profile_ellipse - fsp.v_), rmsh.ds_poly):.{io.number_of_decimals}e}", \
        fieldnames[7]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(ufl.as_tensor(rpam.parameters['mu'] * ela.G(fsp.u_n_1)[j, 0] * (fsp.V[i].dx(j)), (i))), rmsh.ds_r):.{io.number_of_decimals}e}", \
        fieldnames[8]: \
            f"{msh.abs_wrt_measure(ela.G(fsp.u_n_1)[j, i] * bgeo.facet_normal[j] * ela.G(fsp.u_n_1)[l, i] * (fsp.phi.dx(l)), rmsh.ds_l_tb_poly):.{io.number_of_decimals}e}", \
        fieldnames[9]: \
            f"{msh.abs_wrt_measure(fsp.phi, rmsh.ds_r):.{io.number_of_decimals}e}", \
        }])

    csvfile.flush()
