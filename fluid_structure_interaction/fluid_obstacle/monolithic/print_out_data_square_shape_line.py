'''
this module prints some useful data (mean displacement of the elastic body, pressure at the interface ... ) to monitor the time iteration
'''

import importlib
from fenics import *
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import physics.elasticity as ela
import mesh_quality as msh_qu
import mesh.utils as msh
import parameters.read.solution as rpam
import physics.fluid_mechanics as flu
import switch_problem as swi

fi = importlib.import_module(swi.fi)
fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k = ufl.indices(3)


def f_fluid():
    return as_tensor(ela.detF(fsp.u_n(vp.sub_mesh_0_label)) * bgeo.facet_normal[0](vp.sub_mesh_0_label)[k] * ela.G(fsp.u_n(vp.sub_mesh_1_label))[k, j] * flu.sigma_ale(fsp.v_n(vp.sub_mesh_1_label), fsp.sigma_n(vp.sub_mesh_1_label), fsp.u_n(vp.sub_mesh_1_label), rpam.parameters['mu_square'])[i, j], (i))

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
        fi.fieldnames_data[6]: \
            f"{msh.average_wrt_measure(geo.ufl_norm(f_fluid()), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[7]: \
            f"{msh.average_wrt_measure(geo.ufl_norm(vp.f_shape(fsp.c_n(vp.sub_mesh_1_label), msh.average(fsp.u_n), msh.average(fsp.mu_n), bgeo.facet_normal[0](vp.sub_mesh_0_label))), rmsh.ds_mesh[0]['dS_shape']):.{rpam.parameters['print_out_digits']}e}"
        }])

    fi.csvfile_data.flush()
