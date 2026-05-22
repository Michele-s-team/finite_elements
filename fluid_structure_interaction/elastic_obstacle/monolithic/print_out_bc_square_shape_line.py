import importlib
from fenics import *
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import files as fi
import physics.elasticity as ela
import physics.fluid_mechanics as flu
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l, m, n = ufl.indices(6)

# this function prints out the residuals of BCs
def print_bcs():

    fi.writer_bcs.writerows([{
        fi.fieldnames_bcs[0]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_l), rmsh.ds_mesh[0]['ds_l']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[1]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_tb), rmsh.ds_mesh[0]['ds_tb']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[2]: \
            f"{msh.abs_wrt_measure(sqrt((fsp.v_n(vp.sub_mesh_1_label)[i] - msh.average(fsp.u_dot_n[i])) * (fsp.v_n(vp.sub_mesh_1_label)[i] - msh.average(fsp.u_dot_n[i]))), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[3]: \
            f"{msh.abs_wrt_measure(sqrt(flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_fluid'])[i, 0] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_fluid'])[i, 0]), rmsh.ds_mesh[0]['ds_r']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[4]: \
            f"{msh.abs_wrt_measure(sqrt(fsp.sigma_n**2), rmsh.ds_mesh[0]['ds_r']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[5]: \
            f"{msh.abs_wrt_measure( ( bgeo.facet_normal[0](vp.sub_mesh_0_label)[j] * ela.N(fsp.u_n(vp.sub_mesh_0_label), rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, j] - ( flu.sigma_ale(fsp.v_n(vp.sub_mesh_1_label), fsp.sigma_n(vp.sub_mesh_1_label), fsp.u_n(vp.sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, j] * msh.average(ela.detF(fsp.u_n) * ela.G(fsp.u_n)[k, j] ) * bgeo.facet_normal[0](vp.sub_mesh_0_label)[k] ) ) * ( bgeo.facet_normal[0](vp.sub_mesh_0_label)[l] * ela.N(fsp.u_n(vp.sub_mesh_0_label), rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, l] - ( flu.sigma_ale(fsp.v_n(vp.sub_mesh_1_label), fsp.sigma_n(vp.sub_mesh_1_label), fsp.u_n(vp.sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, l] * msh.average(ela.detF(fsp.u_n) * ela.G(fsp.u_n)[m, l] ) *  bgeo.facet_normal[0](vp.sub_mesh_0_label)[m] ) ), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[6]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n), rmsh.ds_mesh[0]['ds']):.{rpam.parameters['print_out_digits']}e}", \
        fi.fieldnames_bcs[7]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_bcs[8]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_dot_n), rmsh.ds_mesh[0]['ds']):.{rpam.parameters['print_out_digits']}e}", \
        fi.fieldnames_bcs[9]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}"
        }])

    fi.csvfile_bcs.flush()
