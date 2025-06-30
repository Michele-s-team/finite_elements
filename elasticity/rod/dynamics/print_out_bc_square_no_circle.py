import colorama as col
import csv
from fenics import *
import importlib
import os
import ufl as ufl

import boundary_geometry as bgeo
import elasticity as ela
import files as fi
import input_output as io
import mesh as msh
import read_parameters as rpam
import solution_paths as solpath

import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

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

    fi.xdmffile_u.write( u_n_output, t )
    fi.xdmffile_v.write( v_n_output, t )

    io.full_print(u_n_output, 'u_n_' + str(step+1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  rmsh.lmsh.mesh, 'vector')
    io.full_print(v_n_output, 'v_n_' + str(step+1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  rmsh.lmsh.mesh, 'vector')

