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
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l, m, n = ufl.indices(6)

# create the path for the csv file if it does not exist
filename_bcs = os.path.join(rarg.args.output_directory, 'bcs.csv')
os.makedirs(os.path.dirname(filename_bcs), exist_ok=True)

csvfile = open(filename_bcs, 'a', newline='')
fieldnames = [ \
    '<<|v^n - v_l|^2>>_{partial Omega l}', \
    '<<|v^n - v_tb|^2>>_{partial Omega tb}',\
    '<<|v^{n square} - average(u_dot_n)|^2>>_{partial Omega ellipse}',\
    '<<varsigma_{i 1} varsigma_{i 1}>>_{partial Omega r}',\
    '<<varsigma^2>>_{partial Omega r}',\
    '<<|u^n|^2>>_{partial Omega circle}', \
    '<<(nu_j P_{ij} - vasigma_{ij} |F| G_{kj} nu_k) (nu_j P_{il} - vasigma_{il} |F| G_{ml} nu_m)>>_{partial Omega circle}', \
    '<<|u^n|^2>>_{partial Omega square}', \
    '<<[u^n_i]_j [u^n_i]_j>>_{partial Omega ellipse}',\
    '<<|\dot{u}^n|^2>>_{partial Omega square}', \
    '<<[\dot{u}^n_i]_j [\dot{u}^n_i]_j>>_{partial Omega ellipse}'
    ]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()



# this function prints out the residuals of BCs
def print_bcs():

    writer.writerows([{
        fieldnames[0]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_l), rmsh.ds_l):.{io.number_of_decimals}e}",\
        fieldnames[1]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.v_n - fsp.v_tb), rmsh.ds_tb):.{io.number_of_decimals}e}",\
        fieldnames[2]: \
            f"{msh.abs_wrt_measure(sqrt((fsp.v_n(vp.sub_mesh_1_label)[i] - msh.average(fsp.u_dot_n[i])) * (fsp.v_n(vp.sub_mesh_1_label)[i] - msh.average(fsp.u_dot_n[i]))), rmsh.dS_ellipse):.{io.number_of_decimals}e}",\
        fieldnames[3]: \
            f"{msh.abs_wrt_measure(sqrt(flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_fluid'])[i, 0] * flu.sigma_ale(fsp.v_n, fsp.sigma_n, fsp.u_n, rpam.parameters['mu_fluid'])[i, 0]), rmsh.ds_r):.{io.number_of_decimals}e}",\
        fieldnames[4]: \
            f"{msh.abs_wrt_measure(sqrt(fsp.sigma_n**2), rmsh.ds_r):.{io.number_of_decimals}e}",\
        fieldnames[5]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n), rmsh.ds_circle):.{io.number_of_decimals}e}",\
        fieldnames[6]: \
            f"{msh.abs_wrt_measure( ( bgeo.facet_normal[0](vp.sub_mesh_0_label)[j] * ela.N(fsp.u_n(vp.sub_mesh_0_label), rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, j] - ( flu.sigma_ale(fsp.v_n(vp.sub_mesh_1_label), fsp.sigma_n(vp.sub_mesh_1_label), fsp.u_n(vp.sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, j] * msh.average(ela.detF(fsp.u_n) * ela.G(fsp.u_n)[k, j] ) * bgeo.facet_normal[0](vp.sub_mesh_0_label)[k] ) ) * ( bgeo.facet_normal[0](vp.sub_mesh_0_label)[l] * ela.N(fsp.u_n(vp.sub_mesh_0_label), rpam.parameters['K_elastic'], rpam.parameters['mu_elastic'])[i, l] - ( flu.sigma_ale(fsp.v_n(vp.sub_mesh_1_label), fsp.sigma_n(vp.sub_mesh_1_label), fsp.u_n(vp.sub_mesh_1_label), rpam.parameters['mu_fluid'])[i, l] * msh.average(ela.detF(fsp.u_n) * ela.G(fsp.u_n)[m, l] ) *  bgeo.facet_normal[0](vp.sub_mesh_0_label)[m] ) ), rmsh.dS_ellipse):.{io.number_of_decimals}e}",\
        fieldnames[7]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_n), rmsh.ds_lrtb):.{io.number_of_decimals}e}", \
        fieldnames[8]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_n[i], bgeo.facet_normal[0])[j]), rmsh.dS_ellipse):.{io.number_of_decimals}e}",\
        fieldnames[9]: \
            f"{msh.abs_wrt_measure(geo.ufl_norm(fsp.u_dot_n), rmsh.ds_lrtb):.{io.number_of_decimals}e}", \
        fieldnames[10]: \
            f"{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.u_dot_n[i], bgeo.facet_normal[0])[j]), rmsh.dS_ellipse):.{io.number_of_decimals}e}"
        }])

    csvfile.flush()
