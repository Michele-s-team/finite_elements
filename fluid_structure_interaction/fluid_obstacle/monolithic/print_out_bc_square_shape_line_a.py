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
    return as_tensor(flu.sigma_ale(msh.side(fsp.v_n, rmsh.I_sub_mesh_0_0), msh.side(fsp.sigma_n, rmsh.I_sub_mesh_0_0), msh.side(fsp.u_n, rmsh.I_sub_mesh_0_0), rpam.parameters['mu_shape'])[i, j] * ela.G(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_0))[k, j] * msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_0)[k] - ( msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_0)[k] * ela.G(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1))[k, j] * flu.sigma_ale(msh.side(fsp.v_n, rmsh.I_sub_mesh_0_1), msh.side(fsp.sigma_n, rmsh.I_sub_mesh_0_1), msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), rpam.parameters['mu_square'])[i, j] \
    + 1.0/ela.detF(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_0)) * vp.f_shape(msh.side(fsp.c_n, rmsh.I_sub_mesh_0_1), msh.average(fsp.u_n), msh.average(fsp.mu_n), msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_0))[i] ), (i))


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
            f"{msh.abs_wrt_measure(( ( ( msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1)[i] - msh.side(fsp.u_n_1, rmsh.I_sub_mesh_0_1)[i] ) *  bgeo.n_cur(msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_0), msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), msh.side(fsp.dyds, rmsh.I_sub_mesh_0_1))[i] ) - ( ( msh.side(fsp.v_n, rmsh.I_sub_mesh_0_1)[i] * vp.dt * bgeo.n_cur(msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_0), msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), msh.side(fsp.dyds, rmsh.I_sub_mesh_0_1))[i] ) ) ), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[8]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_dot_n), rmsh.ds_mesh[0]['ds']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[9]: \
            f"{msh.abs_wrt_measure( ( msh.side(fsp.u_dot_n, rmsh.I_sub_mesh_0_1)[i] * bgeo.n_cur(msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_0), msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), msh.side(fsp.dyds, rmsh.I_sub_mesh_0_1))[i] - msh.side(fsp.v_n, rmsh.I_sub_mesh_0_1)[i] * bgeo.n_cur(msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_0), msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1), msh.side(fsp.dyds, rmsh.I_sub_mesh_0_1))[i] ), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[10]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[11]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[12]: \
            f"{msh.abs_wrt_measure(ela.G(fsp.u_n)[k, i] * (-bgeo.facet_normal[0][k]) * ( -rpam.parameters['D']*ela.G(fsp.u_n)[j, i]*(fsp.c_n.dx(j)) ), rmsh.ds_mesh[0]['ds']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[13]: \
            f"{msh.abs_wrt_measure(ela.detF(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1)) * ela.G(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1))[k, i] * msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_0)[k] * ( -rpam.parameters['D']*ela.G(msh.side(fsp.u_n, rmsh.I_sub_mesh_0_1))[j, i]*(msh.side(fsp.c_n, rmsh.I_sub_mesh_0_1).dx(j)) ) - rpam.parameters['kappa'], rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}"
        }])

    fi.csvfile_bcs.flush()
