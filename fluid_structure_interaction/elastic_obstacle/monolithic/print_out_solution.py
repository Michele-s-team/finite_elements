import csv
from fenics import *
import importlib

import files as fi
import function_spaces as fsp
import input_output as io
import os
import parameters.read.solution as rpam
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi


rmsh = importlib.import_module(swi.rmsh)


# create the path for the csv file if it does not exist
filename_theta_omega = rarg.args.output_directory + '/theta_omega.csv'
os.makedirs(os.path.dirname(filename_theta_omega), exist_ok=True)

csvfile = open(filename_theta_omega, 'a', newline='')
fieldnames = [ \
    "theta", \
    "omega", \
    ]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()


def print_solution(t, step, dt):

    #1 unpack the mixed field 

    v__dummy, phi_dummy, v_n_dummy, u_n_dummy, u_dot_n_dummy = fsp.psi.split( deepcopy=True )


    #2 write to xdmf files

    fi.xdmffile_v_n.write(v_n_dummy, t)
    fi.xdmffile_v_.write(v__dummy, t)

    fi.xdmffile_sigma_n_12.write(fsp.sigma_n_12, t - dt / 2.0)
    fi.xdmffile_phi.write(phi_dummy, t)

    fi.xdmffile_u_n.write(u_n_dummy, t)
    fi.xdmffile_u_dot_n.write(u_dot_n_dummy, t)


    # 3 write snapshots
    io.full_print(v__dummy, 'v_bar_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)
    io.full_print(v_n_dummy, 'v_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)
    
    io.full_print(fsp.sigma_n_12, 'sigma_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)
    io.full_print(phi_dummy, 'phi_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)
    
    io.full_print(u_n_dummy, 'u_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)
    io.full_print(u_dot_n_dummy, 'u_dot_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)


# print solution metadata
io.write_parameters_to_csv_file(os.path.join(solpath.csv_files_path, "metadata.csv"), rpam.parameters)

