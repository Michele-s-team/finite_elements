from fenics import *

import csv
import files as fi
import function_spaces as fsp
import function as fu
import input_output as io
import load_mesh as lmsh
import mesh as msh
import os
import solution_paths as solpath

import runtime_arguments as rarg

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

    u_el_n_output, u_el_dot_n_output = fsp.psi_el.split(deepcopy=True)


    # 1) print the solution for the elastic problem
    io.full_print(u_el_n_output, 'u_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path,
                  lmsh.sub_meshes[0], 'vector')
    io.full_print(u_el_dot_n_output, 'u_dot_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path,
                  lmsh.sub_meshes[0], 'vector')

    # include the snapshot in xdmf files
    fi.xdmffile_u_n.write(u_el_n_output, t)
    fi.xdmffile_u_dot_n.write(u_el_dot_n_output, t)


    '''
    # 2) print the solution for the mesh problem
    io.full_print(fsp.u_el_n, 'u_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path,
                  lmsh.mesh, 'vector')
    io.full_print(fsp.u_el_dot_n, 'u_dot_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path,
                  lmsh.mesh, 'vector')

    # include the snapshot in xdmf files
    fi.xdmffile_u_n.write(fsp.u_el_n, t)
    fi.xdmffile_u_dot_n.write(fsp.u_el_dot_n, t)

    # Write the deformed mesh to file
    deformed_mesh = msh.deform_mesh(lmsh.mesh, fsp.u_el_n)
    with XDMFFile(solpath.snapshots_path + 'mesh_n_' + str(step) + '.xdmf') as xdmf:
        xdmf.write(deformed_mesh)
    io.print_mesh_vertices_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'vertex_mesh_n_' + str(step) + '.csv')
    io.print_mesh_lines_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'line_mesh_n_' + str(step) + '.csv')


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

    # include the snapshot in xdmf files
    fi.xdmffile_v_n.write(fsp.v_n, t)
    fi.xdmffile_v_.write(fsp.v_, t)
    fi.xdmffile_sigma.write(fsp.sigma_n_12, t - dt / 2.0)
    fi.xdmffile_phi.write(fsp.phi, t)


    io.full_print_deformed(fsp.v_, fsp.u_el_n, 'v_bar_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, lmsh.mesh, 'vector')
    io.full_print_deformed(fsp.v_n, fsp.u_el_n, 'v_n_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, lmsh.mesh, 'vector')
    io.full_print_deformed(fsp.sigma_n_12, fsp.u_el_n, 'sigma_n_12_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, lmsh.mesh, 'scalar')
    io.full_print_deformed(fsp.phi, fsp.u_el_n, 'phi_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, lmsh.mesh, 'scalar')

    '''
