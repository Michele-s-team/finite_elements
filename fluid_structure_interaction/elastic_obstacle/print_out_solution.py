from fenics import *

import csv

import elasticity as ela
import files as fi
import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import os
import parameters.read.solution as rpam
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


# print the solution for the elastic problem
def print_solution_el(t, step):
    
    u_el_n_output, u_el_dot_n_output = fsp.psi_el.split(deepcopy=True)

    io.full_print(u_el_n_output, 'u_el_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path,
                  lmsh.sub_meshes[0], 'vector')
    io.full_print(u_el_dot_n_output, 'u_el_dot_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path,
                  lmsh.sub_meshes[0], 'vector')
    
    # print the determinant of the gradient of the deformation field
    io.full_print_deformed(
        project(ela.detF(u_el_n_output), fsp.Q_det_F_u_el),
        u_el_n_output, 'det_F_u_el_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, 'scalar')


    # include the snapshot in xdmf files
    fi.xdmffile_u_el_n.write(u_el_n_output, t)
    fi.xdmffile_u_el_dot_n.write(u_el_dot_n_output, t)

    # Write the deformed mesh of el problem to file
    deformed_mesh = msh.deform_mesh(lmsh.sub_meshes[0], u_el_n_output)
    with XDMFFile(solpath.snapshots_path + 'mesh_el_n_' + str(step) + '.xdmf') as xdmf:
        xdmf.write(deformed_mesh)
    io.print_mesh_vertices_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'vertex_mesh_el_n_' + str(step) + '.csv')
    io.print_mesh_lines_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'line_mesh_el_n_' + str(step) + '.csv')



# print the solution for the mesh problem
def print_solution_msh(t, step):
    io.full_print(fsp.u_msh_n, 'u_msh_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path,
                  lmsh.sub_meshes[1], 'vector')
    io.full_print(fsp.u_msh_dot_n, 'u_msh_dot_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path,
                  lmsh.sub_meshes[1], 'vector')

    # include the snapshot in xdmf files
    fi.xdmffile_u_msh_n.write(fsp.u_msh_n, t)
    fi.xdmffile_u_msh_dot_n.write(fsp.u_msh_dot_n, t)

    # Write the deformed mesh of msh problem to file
    deformed_mesh = msh.deform_mesh(lmsh.sub_meshes[1], fsp.u_msh_n)
    with XDMFFile(solpath.snapshots_path + 'mesh_msh_n_' + str(step) + '.xdmf') as xdmf:
        xdmf.write(deformed_mesh)
    io.print_mesh_vertices_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'vertex_mesh_msh_n_' + str(step) + '.csv')
    io.print_mesh_lines_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'line_mesh_msh_n_' + str(step) + '.csv')


# print the solution of the fluid problem
def print_solution_fl(t, step, dt):
    io.full_print(fsp.v_, 'v_bar_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[1], 'vector')
    io.full_print(fsp.v_n, 'v_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[1], 'vector')
    io.full_print(fsp.sigma_n_12, 'sigma_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[1], 'scalar')
    io.full_print(fsp.phi, 'phi_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[1], 'scalar')

    # include the snapshot in xdmf files
    fi.xdmffile_v_n.write(fsp.v_n, t)
    fi.xdmffile_v_.write(fsp.v_, t)
    fi.xdmffile_sigma.write(fsp.sigma_n_12, t - dt / 2.0)
    fi.xdmffile_phi.write(fsp.phi, t)

    io.full_print_deformed(fsp.v_, fsp.u_msh_n, 'v_bar_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, 'vector')
    io.full_print_deformed(fsp.v_n, fsp.u_msh_n, 'v_n_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, 'vector')
    io.full_print_deformed(fsp.sigma_n_12, fsp.u_msh_n, 'sigma_n_12_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, 'scalar')
    io.full_print_deformed(fsp.phi, fsp.u_msh_n, 'phi_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, 'scalar')


def print_solution(t, step, dt):
    print_solution_el(t, step)
    print_solution_msh(t, step)
    print_solution_fl(t, step, dt)

    # print solution metadata
    io.write_parameters_to_csv_file(solpath.csv_files_path + "metadata.csv", rpam.parameters)

