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
        }])

    fi.csvfile_bcs.flush()
