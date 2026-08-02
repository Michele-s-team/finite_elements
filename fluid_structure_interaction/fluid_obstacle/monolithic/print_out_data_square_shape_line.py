'''
this module prints some useful data (mean displacement of the elastic body, pressure at the interface ... ) to monitor the time iteration
'''

import importlib
from fenics import *
import ufl as ufl

import physics.elasticity as ela
import mesh_quality as msh_qu
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

fi = importlib.import_module(swi.fi)
fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k = ufl.indices(3)


def print_data(step):

    v_n_dummy, _, u_n_dummy, _, _, _, _ = fsp.psi.split( deepcopy=True )

    fi.writer_data.writerows([{
        fi.fieldnames_data[0]: \
            step,\
        fi.fieldnames_data[1]: \
            f"{msh.average_wrt_measure(u_n_dummy[1], rmsh.dx_mesh[0]['dx_shape']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[2]: \
            f"{[msh.average_wrt_measure(fsp.y[i] + u_n_dummy[i], rmsh.dx_mesh[0]['dx_shape']) for i in range(2)]}",\
        fi.fieldnames_data[3]: \
            f"{assemble(ela.detF(u_n_dummy) * rmsh.dx_mesh[0]['dx_shape']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[4]: \
            f"{msh_qu.quality:.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[5]: \
            f"{[assemble(v_n_dummy[i] * ela.detF(u_n_dummy) * rmsh.dx_mesh[0]['dx_shape']) / assemble(ela.detF(u_n_dummy) * rmsh.dx_mesh[0]['dx_shape']) for i in range(len(v_n_dummy))]}",\

        }])

    fi.csvfile_data.flush()
