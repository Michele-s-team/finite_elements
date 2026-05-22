'''
this module prints some useful data (mean displacement of the elastic body, pressure at the interface ... ) to monitor the time iteration
'''

import csv
import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
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

# create the path for the csv file if it does not exist
filename_data = os.path.join(rarg.args.output_directory, 'data.csv')
os.makedirs(os.path.dirname(filename_data), exist_ok=True)

csvfile_data = open(filename_data, 'a', newline='')
fieldnames_data = [ \
    'step',
    '<<|u_n|^2>>_{partial Omega ellipse}',
    '<<sigma_n^2>>_{partial Omega ellipse}',
    '<<varsigma_no_pressure_{ij} varsigma_no_pressure_{ij}>>_{partial Omega ellipse}',
    'mesh_quality',
    'int_shape psi dx'
    ]
writer_data = csv.DictWriter(csvfile_data, fieldnames=fieldnames_data)
writer_data.writeheader()


def print_data(step):

    writer_data.writerows([{
        fieldnames_data[0]: \
            step,
        fieldnames_data[1]: \
            sqrt(assemble(msh.average(fsp.u_n[i]*fsp.u_n[i]) * rmsh.ds_mesh[0]['dS_shape'])),
        fieldnames_data[2]: \
            sqrt(assemble((fsp.sigma_n(vp.sub_mesh_1_label))**2 * rmsh.ds_mesh[0]['dS_shape'])),
        fieldnames_data[3]: \
            sqrt(assemble(flu.sigma_ale_no_pressure(fsp.v_n(vp.sub_mesh_1_label), Constant(0), fsp.u_n(vp.sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, k] * flu.sigma_ale_no_pressure(fsp.v_n(vp.sub_mesh_1_label), Constant(0), fsp.u_n(vp.sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, k] * rmsh.ds_mesh[0]['dS_shape'])),
        fieldnames_data[4]: \
            msh_qu.quality,
        fieldnames_data[5]: \
            assemble(ela.psi(fsp.u_n, rpam.parameters['K_elastic'], rpam.parameters['mu_elastic']) * rmsh.dx_mesh[0]['dx_shape']),
        }])

    csvfile_data.flush()
