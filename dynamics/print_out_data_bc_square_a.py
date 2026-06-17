'''
this module prints some useful data  to monitor the time iteration
'''

import importlib
from fenics import *
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import parameters.read.solution as rpam
import switch_problem as swi

import files as fi
import function_spaces as fsp

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)



i, j, k = ufl.indices(3)


def print_data(step):

    fi.writer_data.writerows([{
        fi.fieldnames_data[0]: \
            f"{step:.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[1]: \
            f"{assemble((fsp.v_bar[i] * geo.g( fsp.omega_n_12 )[i, j] * (bgeo.n_circle( fsp.omega_n_12 ))[j])**2 * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.parameters["c_r"] ) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_circle):.{rpam.parameters['print_out_digits']}e}"      
        }])

    fi.csvfile_data.flush()
