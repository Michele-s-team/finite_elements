import csv
import files as fi
import function_spaces as fsp
import input_output as io
import load_mesh as lmsh
import os
import solution_paths as solpath

import runtime_arguments as rarg

# create the path for the csv file if it does not exist
filename_theta_omega = rarg.args.output_directory + '/theta_omega.csv'
os.makedirs(os.path.dirname(filename_theta_omega), exist_ok=True)

csvfile = open(filename_theta_omega, 'a', newline='')
fieldnames = [ \
    'theta', \
    'omega', \
    ]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()


def print_solution(t, step, dt):
    # 1) print theta and omega
    writer.writerows([{ \
        fieldnames[0]: \
            fsp.theta_n, \
        fieldnames[1]: \
            fsp.omega_n
    }])
    csvfile.flush()


    # 2) print the solution for the mesh problem
    io.full_print(fsp.u_n, 'u_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path,
                  lmsh.mesh, 'vector')
    io.full_print(fsp.u_dot_n, 'u_dot_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path,
                  lmsh.mesh, 'vector')

    # print the solution for the fluid problem
    # include the snapshot in xdmf files
    fi.xdmffile_v_n.write(fsp.v_n, t)
    fi.xdmffile_v_.write(fsp.v_, t)
    fi.xdmffile_sigma.write(fsp.sigma_n_12, t - dt / 2.0)
    fi.xdmffile_phi.write(fsp.phi, t)

    # 3) print the solution of the fluid problem
    io.full_print(fsp.v_, 'v_bar_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.mesh, 'vector')
    io.full_print(fsp.v_n, 'v_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.mesh, 'vector')
    io.full_print(fsp.sigma_n_12, 'sigma_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.mesh, 'scalar')
    io.full_print(fsp.phi, 'phi_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.mesh, 'scalar')
