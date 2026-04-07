from fenics import *

import csv
import importlib

import elasticity as ela
import files as fi
import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import os
import parameters.read.solution as rpam
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

dt = rpam.parameters['T'] / rpam.parameters['N']


# create the path for the mesh csv file if it does not exist
remesh_filename = os.path.join(rarg.args.output_directory, 'remesh.csv')
os.makedirs(os.path.dirname(remesh_filename), exist_ok=True)

remesh_csvfile = open(remesh_filename, 'a', newline='')
remesh_fieldnames = ["remesh_step",  "mesh_quality_before_remesh"]
remesh_writer = csv.DictWriter(remesh_csvfile, fieldnames=remesh_fieldnames)
remesh_writer.writeheader()


data_filename = os.path.join(rarg.args.output_directory, 'data.csv')
os.makedirs(os.path.dirname(data_filename), exist_ok=True)

data_csvfile = open(data_filename, 'a', newline='')
data_fieldnames = ["step",  "mesh_quality"]
data_writer = csv.DictWriter(data_csvfile, fieldnames=data_fieldnames)
data_writer.writeheader()


def print_remesh(step,  mesh_quality_before_remesh):

    remesh_writer.writerows([{ \
        remesh_fieldnames[0]: \
            step, 
        remesh_fieldnames[1]: \
            mesh_quality_before_remesh, 
    }])
    remesh_csvfile.flush()

def print_data(step,  mesh_quality):

    data_writer.writerows([{ \
        data_fieldnames[0]: \
            step, 
        data_fieldnames[1]: \
            mesh_quality, 
    }])
    data_csvfile.flush()


# print the solution for I
def print_solution_I(t, step):

    nu_n_12_output, dpsi_n_12_output = fsp.nu_and_dpsi_n_12.split(deepcopy=True)

    io.full_print(fsp.ys, 'ys_' + str(step), \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

    io.full_print(fsp.U_n_12, 'U_n_12_' + str(step), \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(project(fsp.ys + fsp.U_n_12, fsp.Q_U), 'X_n_12_' + str(step), \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

    io.full_print(fsp.U_n_12_smooth, 'U_n_12_sm_' + str(step), \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)


    io.full_print(nu_n_12_output, 'nu_n_12_' + str(step), \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(dpsi_n_12_output, 'dpsi_n_12_' + str(step), \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

    io.full_print(fsp.mu_n_12, 'mu_n_12_' + str(step), \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)


    # include the snapshot in xdmf files
    # fi.xdmffile_U_n_12.write(fsp.U_n_12, t)


# print the solution for D 
def print_solution_D(t, step):
   
    #1 print fields for di    
    io.full_print(fsp.u_n_di, 'u_n_di_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    
    io.full_print(fsp.u_n_di_dot, 'u_n_di_dot_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    
    # fi.xdmffile_u_n_di.write(fsp.u_n_di, t)
    # fi.xdmffile_u_n_di_dot.write(fsp.u_n_di_dot, t)

    # Write the deformed sub_mesh[0][0] to file
    deformed_sub_mesh_0_0 = msh.deform_mesh(lmsh.sub_meshes[0][0], fsp.u_n_di)
    with XDMFFile(solpath.snapshots_path + 'sub_mesh_0_0_msh_n_' + str(step) + '.xdmf') as xdmf:
        xdmf.write(deformed_sub_mesh_0_0)
    io.print_mesh_vertices_to_csv(deformed_sub_mesh_0_0, solpath.snapshots_csv_path + 'vertex_sub_mesh_0_0_msh_n_' + str(step) + '.csv')
    io.print_mesh_lines_to_csv(deformed_sub_mesh_0_0, solpath.snapshots_csv_path + 'line_sub_mesh_0_0_msh_n_' + str(step) + '.csv')

    # print the boundary points of the mesh boundary given by the shape
    msh.sorted_boundary_points(
        rmsh.lmsh.mesh[0], 
        os.path.join(rarg.args.input_directory, f'mesh_{0}'), 
        [rmsh.lmsh.mesh_parameters[0]['shape_id']],
        os.path.join(solpath.snapshots_csv_path, 'boundary_points_id_' + str(rmsh.lmsh.mesh_parameters[0]['shape_id']) + f'_n_{step}.csv'))


    
    #2 print fields for sq
    io.full_print(fsp.u_n_sq, 'u_n_sq_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    
    io.full_print(fsp.u_n_sq_dot, 'u_n_sq_dot_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    
    # fi.xdmffile_u_n_sq.write(fsp.u_n_sq, t)
    # fi.xdmffile_u_n_sq_dot.write(fsp.u_n_sq_dot, t)

    # Write the deformed sub_mesh[0][1] to file
    deformed_sub_mesh_0_1 = msh.deform_mesh(lmsh.sub_meshes[0][1], fsp.u_n_sq)
    # with XDMFFile(solpath.snapshots_path + 'sub_mesh_0_1_msh_n_' + str(step) + '.xdmf') as xdmf:
    #     xdmf.write(deformed_sub_mesh_0_1)
    io.print_mesh_vertices_to_csv(deformed_sub_mesh_0_1, solpath.snapshots_csv_path + 'vertex_sub_mesh_0_1_msh_n_' + str(step) + '.csv')
    io.print_mesh_lines_to_csv(deformed_sub_mesh_0_1, solpath.snapshots_csv_path + 'line_sub_mesh_0_1_msh_n_' + str(step) + '.csv')

    


# print the solution for disk fluid 
def print_solution_di_fluid(t, step):

    phi_disk_output, omega_disk_output = fsp.phi_omega_disk.split(deepcopy=True)

    # 1 print velocities

    # 1.1 on the reference mesh 
    io.full_print(fsp.v_disk__, 'v_disk__' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.v_disk_n, 'v_disk_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

    # 1.2 on the current mesh
    io.full_print_deformed(fsp.v_disk__, fsp.u_n_di, 'v_disk__' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print_deformed(fsp.v_disk_n, fsp.u_n_di, 'v_disk_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)


    # fi.xdmffile_v_di__.write(fsp.v_disk__, t)
    # fi.xdmffile_v_di_n.write(fsp.v_disk_n, t)



    # 2 print tension 

    # 2.1 on the reference mesh 
    io.full_print(phi_disk_output, 'phi_disk_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(omega_disk_output, 'omega_disk_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.sigma_disk_n_12, 'sigma_disk_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    
    # 2.2 on the current mesh

    io.full_print_deformed(phi_disk_output, fsp.u_n_di, 'phi_disk_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print_deformed(omega_disk_output, fsp.u_n_di, 'omega_disk_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print_deformed(fsp.sigma_disk_n_12, fsp.u_n_di, 'sigma_disk_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    
    # fi.xdmffile_sigma_di_n_12.write(fsp.sigma_disk_n_12, t)


# print the solution for square fluid 
def print_solution_sq_fluid(t, step):

    # 1 print velocities

    # 1.1 on the reference mesh
    io.full_print(fsp.v_square__, 'v_square__' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.v_square_n, 'v_square_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    

    # 1.2 on the current mesh
    
    io.full_print_deformed(fsp.v_square__, fsp.u_n_sq, 'v_square__' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print_deformed(fsp.v_square_n, fsp.u_n_sq, 'v_square_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

    # fi.xdmffile_v_sq__.write(fsp.v_square__, t)
    # fi.xdmffile_v_sq_n.write(fsp.v_square_n, t)



    # 2 print tension 

    # 2.1 on the reference mesh

    io.full_print(fsp.phi_square, 'phi_square_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.sigma_square_n_12, 'sigma_square_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    
    # 2.2 on the current mesh 

    io.full_print_deformed(fsp.phi_square, fsp.u_n_sq, 'phi_square_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print_deformed(fsp.sigma_square_n_12, fsp.u_n_sq, 'sigma_square_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)


    # fi.xdmffile_sigma_sq_n_12.write(fsp.sigma_square_n_12, t)



# print the solution for M
def print_solution_M(t, step):

    # 1 on the reference mesh
    io.full_print(fsp.c_n, 'c_n_' + str(step), \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    
    # 2 on the current mesh
    io.full_print_deformed(fsp.c_n, fsp.u_n_sq, 'c_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    
    # fi.xdmffile_c_n.write(fsp.c_n, t)


# print solution for all sectors
def print_solution(t, step):

    print_solution_I(t, step)
    print_solution_D(t, step)
    print_solution_di_fluid(t, step)
    print_solution_sq_fluid(t, step)
    print_solution_M(t, step)



# print solution metadata
io.write_parameters_to_csv_file(solpath.csv_files_path + "metadata.csv", rpam.parameters)

