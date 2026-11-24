import importlib
from fenics import *
import ufl as ufl

import differential_geometry.manifold.geometry as geo
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

fi = importlib.import_module(swi.fi)
fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l, m, n = ufl.indices(6)

'''
residual in BC (60) in 'Channel flow with moving obstacle'
'''
def bc_residual(i):

    return ( msh.side(fsp.v_n, rmsh.I_sub_mesh_0_1)[i]  - ( msh.average(fsp.omega_n) * vp.dRdtheta(msh.average(fsp.theta_n - Constant(rmsh.parameters['phi'])))[i, j] * ( msh.average(fsp.y)[j] - vp.c()[j] ) ) )

# this function prints out the residuals of BCs
def print_bcs(step):



    fi.writer_bcs.writerows([{
        fi.fieldnames_bcs[0]: \
            step,
        fi.fieldnames_bcs[1]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_l), rmsh.ds_mesh[0]['ds_l']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[2]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_tb), rmsh.ds_mesh[0]['ds_tb']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[3]: \
            f"{msh.abs_wrt_measure( sqrt(bc_residual(i)*bc_residual(i)), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[4]: \
            f"{msh.abs_wrt_measure( fsp.sigma_n, rmsh.ds_mesh[0]['ds_r']):.{rpam.parameters['print_out_digits']}e}"
        }]
        )

    fi.csvfile_bcs.flush()
