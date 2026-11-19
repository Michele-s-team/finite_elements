'''
this module prints some useful data (mean displacement of the elastic body, pressure at the interface ... ) to monitor the time iteration
'''

import importlib
from fenics import *
import ufl as ufl

import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

fi = importlib.import_module(swi.fi)
fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k = ufl.indices(3)


def print_data(step):

    theta_n_dummy, omega_n_dummy, _, _, _, _ = fsp.psi.split( deepcopy=True )

    fi.writer_data.writerows([{
        fi.fieldnames_data[0]: \
            f"{step:.{rpam.parameters['print_out_digits']}e}",
        fi.fieldnames_data[1]: \
            f"{msh.average_wrt_measure(theta_n_dummy, rmsh.dx_mesh[0]['dx']):.{rpam.parameters['print_out_digits']}e}",
        fi.fieldnames_data[2]: \
            f"{msh.average_wrt_measure(omega_n_dummy, rmsh.dx_mesh[0]['dx']):.{rpam.parameters['print_out_digits']}e}"
        }])

    fi.csvfile_data.flush()
