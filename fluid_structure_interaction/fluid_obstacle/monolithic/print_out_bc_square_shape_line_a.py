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

i, j, k, l, m, n = ufl.indices(6)

# this function prints out the residuals of BCs
def print_bcs(step):

    fi.writer_bcs.writerows([{
        fi.fieldnames_bcs[0]: \
            step,
            fi.fieldnames_bcs[1]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_lrb), rmsh.ds_mesh[0]['ds_lr'] + rmsh.ds_mesh[0]['ds_b']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[2]: \
            f"{msh.abs_wrt_measure( sqrt((bgeo.facet_normal[0][k] * ela.G(fsp.u_n)[k, j] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_square'])[i, j] * ela.detF(fsp.u_n) - fsp.t_t[i]) * (bgeo.facet_normal[0][l] * ela.G(fsp.u_n)[l, m] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_square'])[i, m] * ela.detF(fsp.u_n)-  - fsp.t_t[i])), rmsh.ds_mesh[0]['ds_t']):.{rpam.parameters['print_out_digits']}e}",\
            fi.fieldnames_bcs[3]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
        }])

    fi.csvfile_bcs.flush()
