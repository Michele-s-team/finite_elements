'''
this module prints the ICs (internal conditions) relative to the interior facets of the mesh
'''

import importlib
from fenics import *
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

fi = importlib.import_module(swi.fi)
fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l, m, n = ufl.indices(6)


# this function prints out the residuals of BCs
def print_ics(step):

    fi.writer_ics.writerows([{
        fi.fieldnames_ics[0]: \
            step,\
        fi.fieldnames_ics[1]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.v_n[i], bgeo.facet_normal[0])[j]), rmsh.ds_mesh[0]['dS_I_shape']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_ics[2]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.sigma_n, bgeo.facet_normal[0])[i] * msh.jump(fsp.sigma_n, bgeo.facet_normal[0])[i]), rmsh.ds_mesh[0]['dS_I_shape']):.{rpam.parameters['print_out_digits']}e}"
        }])

    fi.csvfile_ics.flush()
