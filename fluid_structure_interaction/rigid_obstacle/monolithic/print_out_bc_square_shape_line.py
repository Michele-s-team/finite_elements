import importlib
from fenics import *
import ufl as ufl

import differential_geometry.manifold.geometry as geo
import mesh.utils as msh
import parameters.read.solution as rpam
import physics.fluid_mechanics as flu
import switch_problem as swi

fi = importlib.import_module(swi.fi)
fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l, m, n = ufl.indices(6)

'''
residual of BC (60) in 'Channel flow with moving obstacle'
'''
def bc_v_residual(i):

    return ( msh.side(fsp.v_n, rmsh.I_sub_mesh_0_1)[i]  - ( msh.average(fsp.omega_n) * vp.dRdtheta(msh.average(fsp.theta_n - Constant(rmsh.parameters['phi'])))[i, j] * ( msh.average(fsp.y)[j] - vp.c()[j] ) ) )


'''
residual of BC (100) in 'Channel flow with moving obstacle'
'''
def bc_u_residual(i):

    return ( ( msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1)[i] - ( (vp.c())[i] - msh.average(fsp.y)[i] + vp.R(msh.average(fsp.theta_n - Constant(rmsh.parameters['phi'])))[i, j] * ( msh.average(fsp.y)[j] - (vp.c())[j] ) ) ) )

# this function prints out the residuals of BCs
def print_bcs(step):



    fi.writer_bcs.writerows([{
        fi.fieldnames_bcs[0]: \
            int(step),
        fi.fieldnames_bcs[1]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_l), rmsh.ds_mesh[0]['ds_l']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[2]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_tb), rmsh.ds_mesh[0]['ds_tb']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[3]: \
            f"{msh.abs_wrt_measure( sqrt(bc_v_residual(i)*bc_v_residual(i)), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[4]: \
            f"{msh.abs_wrt_measure( fsp.sigma_n, rmsh.ds_mesh[0]['ds_r']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[5]: \
            f"{msh.abs_wrt_measure( geo.ufl_norm(as_tensor(flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu'])[i, 0], (i))), rmsh.ds_mesh[0]['ds_r']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[6]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n), rmsh.ds_mesh[0]['ds']):.{rpam.parameters['print_out_digits']}e}", \
        fi.fieldnames_bcs[7]: \
            f"{msh.abs_wrt_measure( sqrt(bc_u_residual(i)*bc_u_residual(i)) , rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}", \
        }]
        )

    fi.csvfile_bcs.flush()
