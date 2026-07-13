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

    v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy, c_n_dummy, mu_n_dummy, grad_u_n_dummy = fsp.psi.split( deepcopy=True )


    u_n_y = u_n_dummy.sub(1, deepcopy=True)
    u_n_y_min = u_n_y.vector().min()
    u_n_y_max = u_n_y.vector().max()

    fi.writer_data.writerows([{
        fi.fieldnames_data[0]: \
            step,\
        fi.fieldnames_data[1]: \
            f"{u_n_y_min:.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[2]: \
            f"{u_n_y_max:.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[3]: \
            f"{msh.average_wrt_measure(u_n_dummy[1], rmsh.dx_mesh[0]['dx_shape']):.{rpam.parameters['print_out_digits']}e}"
        }])

    fi.csvfile_data.flush()
