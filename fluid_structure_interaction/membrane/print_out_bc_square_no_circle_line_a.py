import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi

fi = importlib.import_module(swi.fi)
rmsh = importlib.import_module(swi.rmsh)
vp_mesh = importlib.import_module(swi.vp_mesh)
vp_fluid = importlib.import_module(swi.vp_fluid)

i, j, k, l, alpha, beta, gamma = ufl.indices(7)



# this method prints out the residuals of BCs for all sectors
def print_bcs(step):
    
    fi.writer_bcs.writerows([{

        # 1. membrane problem
        fi.fieldnames_bcs[0]: \
            step, 
        fi.fieldnames_bcs[1]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_bar - fsp.v_bar_l), rmsh.ds_sub_mesh[1]['ds_l']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[2]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_bar - fsp.v_bar_r), rmsh.ds_sub_mesh[1]['ds_r']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[3]: \
            f"{msh.abs_wrt_measure(fsp.w_bar, rmsh.ds_sub_mesh[1]['ds_l']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[4]: \
            f"{msh.abs_wrt_measure(fsp.phi, rmsh.ds_sub_mesh[1]['ds_l']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[5]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.U_n_12), rmsh.ds_sub_mesh[1]['ds_l']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[6]: \
            f"{msh.abs_wrt_measure(fsp.U_n_12[0], rmsh.ds_sub_mesh[1]['ds_r']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[7]: \
            f"{msh.abs_wrt_measure((bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12, lmsh.sub_meshes[1]))[i] * (fsp.phi.dx(i)), rmsh.ds_sub_mesh[1]['ds']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[8]: \
            f"{msh.abs_wrt_measure(fsp.w_bar.dx(0), rmsh.ds_sub_mesh[1]['ds_r']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[9]: \
            f"{msh.abs_wrt_measure(fsp.U_n_12[1].dx(0), rmsh.ds_sub_mesh[1]['ds_r']):.{io.number_of_decimals}e}",
            

        # 2. mesh problem
        fi.fieldnames_bcs[10]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n), rmsh.ds_sub_mesh[0]['ds_l'] + rmsh.ds_sub_mesh[0]['ds_b']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[11]: \
            f"{msh.abs_wrt_measure(fsp.u_n[0], rmsh.ds_sub_mesh[0]['ds_r']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[12]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n - fsp.U_n_12_on_mesh), rmsh.ds_sub_mesh[0]['ds_t']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[13]: \
            f"{msh.abs_wrt_measure(fsp.u_n[1].dx(0), rmsh.ds_sub_mesh[0]['ds_r']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[14]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_dot_n), rmsh.ds_sub_mesh[0]['ds_l'] + rmsh.ds_sub_mesh[0]['ds_b']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[15]: \
            f"{msh.abs_wrt_measure(fsp.u_dot_n[0], rmsh.ds_sub_mesh[0]['ds_r']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[16]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_dot_n - fsp.U_dot_n_12_on_mesh), rmsh.ds_sub_mesh[0]['ds_t']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[17]: \
            f"{msh.abs_wrt_measure(fsp.u_dot_n[1].dx(0), rmsh.ds_sub_mesh[0]['ds_r']):.{io.number_of_decimals}e}",

            
        # 3. fluid problem
        fi.fieldnames_bcs[18]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_fl_bar - fsp.v_fl_bar_b), rmsh.ds_sub_mesh[0]['ds_b']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[19]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_fl_bar), rmsh.ds_sub_mesh[0]['ds_l']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[20]: \
            f"{msh.abs_wrt_measure(abs(fsp.v_fl_bar[0]), rmsh.ds_sub_mesh[0]['ds_r']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[21]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_fl_bar - fsp.u_dot_n), rmsh.ds_sub_mesh[0]['ds_t']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[22]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.phi_fl), rmsh.ds_sub_mesh[0]['ds_b']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[23]: \
            f"{msh.abs_wrt_measure(ela.G(fsp.u_n_1)[alpha, 0] * (((fsp.v_fl_n_1[1] + fsp.v_fl_bar[1]) / 2.0).dx(alpha)), rmsh.ds_sub_mesh[0]['ds_r']):.{io.number_of_decimals}e}",
        fi.fieldnames_bcs[24]: \
            f"{msh.abs_wrt_measure(ela.G(fsp.u_n_1)[gamma, alpha] * (bgeo.sub_mesh_facet_normal[0])[gamma] * ela.G(fsp.u_n_1)[beta, alpha] * (fsp.phi_fl.dx(beta)), rmsh.ds_sub_mesh[0]['ds_l'] + rmsh.ds_sub_mesh[0]['ds_t']):.{io.number_of_decimals}e}",        
        fi.fieldnames_bcs[25]: \
            f"{msh.abs_wrt_measure(ela.G(fsp.u_n_1)[beta, 0] * (fsp.phi_fl.dx(beta)) , rmsh.ds_sub_mesh[0]['ds_r']):.{io.number_of_decimals}e}"        
            }])

    fi.csvfile_bcs.flush()

