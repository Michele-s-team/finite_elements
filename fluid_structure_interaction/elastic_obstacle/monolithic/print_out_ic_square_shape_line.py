'''
this module prints the ICs (internal conditions) relative to the interior facets of the mesh
'''

import csv
import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import files as fi
import physics.elasticity as ela
import physics.fluid_mechanics as flu
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l, m, n = ufl.indices(6)


# this function prints out the residuals of BCs
def print_ics():

    fi.writer_ics.writerows([{
        fi.fieldnames_ics[0]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_I_square']):.{io.number_of_decimals}e}",
        fi.fieldnames_ics[1]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.sigma_n, bgeo.facet_normal[0])[i] * msh.jump(fsp.sigma_n, bgeo.facet_normal[0])[i]), rmsh.ds_mesh[0]['dS_I_square']):.{io.number_of_decimals}e}",
        fi.fieldnames_ics[2]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_I_shape']):.{io.number_of_decimals}e}",
        fi.fieldnames_ics[3]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_I_square']):.{io.number_of_decimals}e}",
        fi.fieldnames_ics[4]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_I_shape']):.{io.number_of_decimals}e}",
        fi.fieldnames_ics[5]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_I_square']):.{io.number_of_decimals}e}",
        }])

    fi.csvfile_ics.flush()
