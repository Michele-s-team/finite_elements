import colorama as col
import csv
from fenics import *
import importlib
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import elasticity as ela
import files as fi
import function_spaces as fsp
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import solution_paths as solpath

import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

# print mesh metadata
io.write_parameters_to_csv_file(rarg.args.output_directory + "/metadata.csv", rpam.parameters)

# set up printout of the BCs to file
# create the path for the csv file if it does not exist
os.makedirs(os.path.dirname(rarg.args.output_directory + '/bcs.csv'), exist_ok=True)

csvfile_bcs = open((rarg.args.output_directory) + '/bcs.csv', 'a', newline='')
fieldnames_bcs = [ \
    '<<|u^n|^2>>_{partial Omega^y l}', \
    '<<|n_j P_{ij}|^2>>_{partial Omega^l r t b}' \
    ]
writer_bcs = csv.DictWriter(csvfile_bcs, fieldnames=fieldnames_bcs)
writer_bcs.writeheader()


# this function prints out the residuals of BCs
def print_bcs(psi):
    # get the fields from psi
    u_n_output, v_n_output = psi.split(deepcopy=True)

    # write the residual of natural BCs on step 2 to file
    writer_bcs.writerows([{ \
        fieldnames_bcs[0]: \
            f"{msh.abs_wrt_measure(u_n_output[i] * u_n_output[i], rmsh.ds_l):.{io.number_of_decimals}e}", \
        fieldnames_bcs[1]: \
            f"{msh.abs_wrt_measure((bgeo.facet_normal[j] * ela.P(u_n_output, rpam.parameters['K'], rpam.parameters['mu'])[i, j]) * (bgeo.facet_normal[k] * ela.P(u_n_output, rpam.parameters['K'], rpam.parameters['mu'])[i, k]), rmsh.ds_r + rmsh.ds_t + rmsh.ds_b):.{io.number_of_decimals}e}"}])
    csvfile_bcs.flush()


def print_solution(psi, step, t):
    # get the fields from psi
    u_n_output, v_n_output = psi.split(deepcopy=True)


    io.full_print(u_n_output, 'u_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(v_n_output, 'v_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    
    # print the determinant of the gradient of the deformation field
    io.full_print_deformed(
        project(ela.detF(u_n_output), fsp.U_det_F), 
        u_n_output, 'det_F_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
     

    fi.xdmffile_u.write( u_n_output, t )
    fi.xdmffile_v.write( v_n_output, t )

    # Write the deformed mesh to file
    deformed_mesh = msh.deform_mesh(rmsh.lmsh.mesh, u_n_output)
    with XDMFFile(solpath.snapshots_path + 'mesh_n_' + str(step) + '.xdmf') as xdmf:
        xdmf.write(deformed_mesh)
    io.print_mesh_vertices_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'vertex_mesh_n_' + str(step) + '.csv')
    io.print_mesh_lines_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'line_mesh_n_' + str(step) + '.csv')


