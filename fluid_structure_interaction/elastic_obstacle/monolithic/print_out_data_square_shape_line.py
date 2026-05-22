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
            f"{step:.{io.number_of_decimals}e}",
        fi.fieldnames_data[1]: \
            f"{sqrt(assemble(msh.average(fsp.u_n[i]*fsp.u_n[i]) * rmsh.ds_mesh[0]['dS_shape'])):.{io.number_of_decimals}e}",
        fi.fieldnames_data[2]: \
            f"{sqrt(assemble((fsp.sigma_n(vp.sub_mesh_1_label))**2 * rmsh.ds_mesh[0]['dS_shape'])):.{io.number_of_decimals}e}",
        fi.fieldnames_data[3]: \
            f"{sqrt(assemble(flu.sigma_ale_no_pressure(fsp.v_n(vp.sub_mesh_1_label), Constant(0), fsp.u_n(vp.sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, k] * flu.sigma_ale_no_pressure(fsp.v_n(vp.sub_mesh_1_label), Constant(0), fsp.u_n(vp.sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, k] * rmsh.ds_mesh[0]['dS_shape'])):.{io.number_of_decimals}e}",
        fi.fieldnames_data[4]: \
            f"{msh_qu.quality:.{io.number_of_decimals}e}",
        fi.fieldnames_data[5]: \
            f"{assemble(ela.psi(fsp.u_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic']) * rmsh.dx_mesh[0]['dx_shape']):.{io.number_of_decimals}e}",
        }])

    fi.csvfile_data.flush()
