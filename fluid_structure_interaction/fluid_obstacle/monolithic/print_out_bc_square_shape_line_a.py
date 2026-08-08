import importlib
from fenics import *
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import physics.fluid_mechanics as flu
import differential_geometry.manifold.geometry as geo
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

fi = importlib.import_module(swi.fi)
fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l, m, n = ufl.indices(6)

# term related to the BC (108)
def bc_shape():
    return as_tensor(flu.sigma_ale(fsp.v_n(vp.sub_mesh_0_label), fsp.sigma_n(vp.sub_mesh_0_label), fsp.u_n(vp.sub_mesh_0_label), rpam.parameters['mu_shape'])[i, j] * ela.G(fsp.u_n(vp.sub_mesh_0_label))[k, j] * bgeo.facet_normal[0](vp.sub_mesh_0_label)[k] - ( bgeo.facet_normal[0](vp.sub_mesh_0_label)[k] * ela.G(fsp.u_n(vp.sub_mesh_1_label))[k, j] * flu.sigma_ale(fsp.v_n(vp.sub_mesh_1_label), fsp.sigma_n(vp.sub_mesh_1_label), fsp.u_n(vp.sub_mesh_1_label), rpam.parameters['mu_square'])[i, j] \
    + 1.0/ela.detF(fsp.u_n(vp.sub_mesh_0_label)) * vp.f_shape(fsp.c_n(vp.sub_mesh_1_label), msh.average(fsp.u_n), msh.average(fsp.mu_n), bgeo.facet_normal[0](vp.sub_mesh_0_label))[i] ), (i))


# this function prints out the residuals of BCs
def print_bcs(step):

    fi.writer_bcs.writerows([{
        fi.fieldnames_bcs[0]: \
            step,
            fi.fieldnames_bcs[1]: \
            f"{msh.abs_wrt_measure(sqrt( bc_shape()[i] * bc_shape()[i]), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[2]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_lrb), rmsh.ds_mesh[0]['ds_lr'] + rmsh.ds_mesh[0]['ds_b']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[3]: \
            f"{msh.abs_wrt_measure( sqrt((bgeo.facet_normal[0][k] * ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_square'])[i, j] * ela.detF(fsp.u_n) - fsp.t_t[i]) * (bgeo.facet_normal[0][l] * ela.G(fsp.u_n)[l, m] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_square'])[i, m] * ela.detF(fsp.u_n)-  - fsp.t_t[i])), rmsh.ds_mesh[0]['ds_t']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[4]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[5]: \
            f"{msh.abs_wrt_measure(fsp.sigma_n - fsp.sigma_square_t, rmsh.ds_mesh[0]['ds_t']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[6]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n), rmsh.ds_mesh[0]['ds']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[7]: \
            f"{msh.abs_wrt_measure(( ( ( fsp.u_n(vp.sub_mesh_1_label)[i] - fsp.u_n_1(vp.sub_mesh_1_label)[i] ) *  bgeo.n_cur(bgeo.facet_normal[0](vp.sub_mesh_0_label), fsp.u_n(vp.sub_mesh_1_label), fsp.dyds(vp.sub_mesh_1_label))[i] ) - ( ( fsp.v_n(vp.sub_mesh_1_label)[i] * vp.dt * bgeo.n_cur(bgeo.facet_normal[0](vp.sub_mesh_0_label), fsp.u_n(vp.sub_mesh_1_label), fsp.dyds(vp.sub_mesh_1_label))[i] ) ) ), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[8]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_dot_n), rmsh.ds_mesh[0]['ds']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[9]: \
            f"{msh.abs_wrt_measure( ( fsp.u_dot_n(vp.sub_mesh_1_label)[i] * bgeo.n_cur(bgeo.facet_normal[0](vp.sub_mesh_0_label), fsp.u_n(vp.sub_mesh_1_label), fsp.dyds(vp.sub_mesh_1_label))[i] - fsp.v_n(vp.sub_mesh_1_label)[i] * bgeo.n_cur(bgeo.facet_normal[0](vp.sub_mesh_0_label), fsp.u_n(vp.sub_mesh_1_label), fsp.dyds(vp.sub_mesh_1_label))[i] ), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}"
        }])

    fi.csvfile_bcs.flush()
