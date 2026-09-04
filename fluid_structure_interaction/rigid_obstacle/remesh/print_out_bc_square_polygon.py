import csv
import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi

fi = importlib.import_module(swi.fi)
rmsh = importlib.import_module(swi.rmsh)
vp_mesh = importlib.import_module(swi.vp_mesh)
vp_fluid = importlib.import_module(swi.vp_fluid)

i, j, k, l = ufl.indices(4)


# this function prints out the residuals of BCs
def print_bcs(step):

    # write the residual of natural BCs  to file
    fi.writer_bcs.writerows([{
        fi.fieldnames_bcs[0]: \
            step, \
        fi.fieldnames_bcs[1]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_shape - fsp.u_n), rmsh.ds_poly):.{io.number_of_decimals}e}", \
        fi.fieldnames_bcs[2]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_square - fsp.u_n), rmsh.ds_square):.{io.number_of_decimals}e}", \
        fi.fieldnames_bcs[3]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_dot_shape - fsp.u_dot_n), rmsh.ds_poly):.{io.number_of_decimals}e}", \
        fi.fieldnames_bcs[4]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_dot_square - fsp.u_dot_n), rmsh.ds_square):.{io.number_of_decimals}e}", \
        fi.fieldnames_bcs[5]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(vp_fluid.v__profile_l - fsp.v_), rmsh.ds_l):.{io.number_of_decimals}e}", \
        fi.fieldnames_bcs[6]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_), rmsh.ds_tb):.{io.number_of_decimals}e}", \
        fi.fieldnames_bcs[7]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(vp_fluid.v__profile_ellipse - fsp.v_), rmsh.ds_poly):.{io.number_of_decimals}e}", \
        fi.fieldnames_bcs[8]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(ufl.as_tensor(rpam.parameters['mu'] * ela.G(fsp.u_n_1)[j, 0] * (fsp.V[i].dx(j)), (i))), rmsh.ds_r):.{io.number_of_decimals}e}", \
        fi.fieldnames_bcs[9]: \
            f"{msh.abs_wrt_measure(ela.G(fsp.u_n_1)[j, i] * bgeo.facet_normal[j] * ela.G(fsp.u_n_1)[l, i] * (fsp.phi.dx(l)), rmsh.ds_l_tb_poly):.{io.number_of_decimals}e}", \
        fi.fieldnames_bcs[10]: \
            f"{msh.abs_wrt_measure(fsp.phi, rmsh.ds_r):.{io.number_of_decimals}e}", \
        }])

    fi.csvfile_bcs.flush()
