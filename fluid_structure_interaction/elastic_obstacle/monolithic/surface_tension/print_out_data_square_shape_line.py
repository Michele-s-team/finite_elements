'''
this module prints some useful data (mean displacement of the elastic body, pressure at the interface ... ) to monitor the time iteration
'''

import importlib
from fenics import *
import ufl as ufl

import continuation as cont
import decompose_u as dec_u
import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import physics.fluid_mechanics as flu
import mesh_quality as msh_qu
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

fi = importlib.import_module(swi.fi)
fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l, m = ufl.indices(5)


def print_data(step):

    fi.writer_data.writerows([{
        fi.fieldnames_data[0]: \
            f"{step:.{rpam.parameters['print_out_digits']}e}",
        fi.fieldnames_data[1]: \
            f"{sqrt(assemble(msh.average(fsp.u_n[i]*fsp.u_n[i]) * rmsh.ds_mesh[0]['dS_shape'])):.{rpam.parameters['print_out_digits']}e}",
        fi.fieldnames_data[2]: \
            f"{sqrt(assemble((msh.side(fsp.sigma_n, rmsh.I_sub_mesh_0_1))**2 * rmsh.ds_mesh[0]['dS_shape'])):.{rpam.parameters['print_out_digits']}e}",
        fi.fieldnames_data[3]: \
            f"{sqrt(assemble(flu.sigma_ale_no_pressure(msh.side(fsp.v_n, rmsh.I_sub_mesh_0_1), Constant(0), msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), rpam.parameters['mu_fluid'])[i, k] * flu.sigma_ale_no_pressure(msh.side(fsp.v_n, rmsh.I_sub_mesh_0_1), Constant(0), msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), rpam.parameters['mu_fluid'])[i, k] * rmsh.ds_mesh[0]['dS_shape'])):.{rpam.parameters['print_out_digits']}e}",
        fi.fieldnames_data[4]: \
            f"{msh_qu.quality:.{rpam.parameters['print_out_digits']}e}",
        fi.fieldnames_data[5]: \
            f"{assemble(ela.psi(fsp.u_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic']) * rmsh.dx_mesh[0]['dx_shape']):.{rpam.parameters['print_out_digits']}e}",
        fi.fieldnames_data[6]: \
            dec_u.t,
        fi.fieldnames_data[7]: \
            dec_u.c,
        fi.fieldnames_data[8]: \
            dec_u.C,
        fi.fieldnames_data[9]: \
            f"{dec_u.theta:.{rpam.parameters['print_out_digits']}e}",
        fi.fieldnames_data[10]: \
            f"{msh.average_wrt_measure( sqrt( (flu.sigma_ale(msh.side(fsp.v_n, rmsh.I_sub_mesh_0_1), cont.pressure_scale * msh.side(fsp.sigma_n, rmsh.I_sub_mesh_0_1), msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), rpam.parameters['mu_fluid'])[i, j] * msh.average(ela.detF(fsp.u_n) * ela.G(fsp.u_n)[k, j]) * msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_0)[k]) *  (flu.sigma_ale(msh.side(fsp.v_n, rmsh.I_sub_mesh_0_1), cont.pressure_scale * msh.side(fsp.sigma_n, rmsh.I_sub_mesh_0_1), msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), rpam.parameters['mu_fluid'])[i, l] * msh.average(ela.detF(fsp.u_n) * ela.G(fsp.u_n)[m, l]) * msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_0)[m]) ), rmsh.ds_mesh[0]['dS_shape'] ):.{rpam.parameters['print_out_digits']}e}", \
        fi.fieldnames_data[11]: \
            f"{msh.average_wrt_measure( sqrt( ( - 2 * rpam.parameters['sigma'] * msh.average(fsp.mu_n * ela.detF(fsp.u_n) * ela.G(fsp.u_n)[k, i]) * msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_0)[k]) * ( - 2 * rpam.parameters['sigma'] * msh.average(fsp.mu_n * ela.detF(fsp.u_n) * ela.G(fsp.u_n)[l, i]) * msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_0)[l]) ), rmsh.ds_mesh[0]['dS_shape'] ):.{rpam.parameters['print_out_digits']}e}"
        }])

    fi.csvfile_data.flush()
