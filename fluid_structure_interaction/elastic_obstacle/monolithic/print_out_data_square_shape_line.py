'''
this module prints some useful data (mean displacement of the elastic body, pressure at the interface ... ) to monitor the time iteration
'''

import importlib
from fenics import *
import ufl as ufl

import files as fi
import physics.elasticity as ela
import physics.fluid_mechanics as flu
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh_quality as msh_qu
import mesh.utils as msh
import parameters.read.solution as rpam
import physics.elasticity as ela
import runtime_arguments as rarg
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k = ufl.indices(3)


def print_data(step):

    fi.writer_data.writerows([{
        fi.fieldnames_data[0]: \
            step,
        fi.fieldnames_data[1]: \
            sqrt(assemble(msh.average(fsp.u_n[i]*fsp.u_n[i]) * rmsh.ds_mesh[0]['dS_shape'])),
        fi.fieldnames_data[2]: \
            sqrt(assemble((fsp.sigma_n(vp.sub_mesh_1_label))**2 * rmsh.ds_mesh[0]['dS_shape'])),
        fi.fieldnames_data[3]: \
            sqrt(assemble(flu.sigma_ale_no_pressure(fsp.v_n(vp.sub_mesh_1_label), Constant(0), fsp.u_n(vp.sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, k] * flu.sigma_ale_no_pressure(fsp.v_n(vp.sub_mesh_1_label), Constant(0), fsp.u_n(vp.sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, k] * rmsh.ds_mesh[0]['dS_shape'])),
        fi.fieldnames_data[4]: \
            msh_qu.quality,
        fi.fieldnames_data[5]: \
            assemble(ela.psi(fsp.u_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic']) * rmsh.dx_mesh[0]['dx_shape']),
        }])

    fi.csvfile_data.flush()
