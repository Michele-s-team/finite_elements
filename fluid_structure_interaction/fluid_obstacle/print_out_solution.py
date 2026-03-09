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


dt = rpam.parameters['T'] / rpam.parameters['N']



# print the solution for I
def print_solution_I(t, step):

    io.full_print(fsp.U_n_12, 'U_n_12_' + str(step), \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
            lmsh.mesh[1], 'vector')

    # include the snapshot in xdmf files
    fi.xdmffile_U_n_12.write(fsp.U_n_12, t)


# print the solution for D 
def print_solution_D(t, step):
   
    #1 print fields for di    
    io.full_print(fsp.u_n_di, 'u_di_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'vector')
    
    io.full_print(fsp.u_n_di_dot, 'u_di_dot_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'vector')
    
    fi.xdmffile_u_n_di.write(fsp.u_n_di, t)
    fi.xdmffile_u_n_di_dot.write(fsp.u_n_di_dot, t)

    
    # print fields for sq
    io.full_print(fsp.u_n_sq, 'u_sq_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][1], 'vector')
    
    io.full_print(fsp.u_n_sq_dot, 'u_sq_dot_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][1], 'vector')
    
    fi.xdmffile_u_n_sq.write(fsp.u_n_sq, t)
    fi.xdmffile_u_n_sq_dot.write(fsp.u_n_sq_dot, t)

    


# print the solution for disk fluid 
def print_solution_di_fluid(t, step):

    phi_disk_output, omega_disk_output = fsp.phi_omega_disk.split(deepcopy=True)

    # 1 print velocities
    # 1.1 on the reference mesh 
    io.full_print(fsp.v_disk__, 'v_disk__' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'vector')
    io.full_print(fsp.v_disk_n, 'v_disk_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'vector')

    # 1.2 on the current mesh
    io.full_print_deformed(fsp.v_disk__, fsp.u_n_di, 'v_disk__' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, lmsh.sub_meshes[0][0], 'vector')
    io.full_print_deformed(fsp.v_disk_n, fsp.u_n_di, 'v_disk_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, lmsh.sub_meshes[0][0], 'vector')

    fi.xdmffile_v_di__.write(fsp.v_disk__, t)
    fi.xdmffile_v_di_n.write(fsp.v_disk_n, t)


    # 2 print tension 
    # 2.1 on the reference mesh 
    io.full_print(phi_disk_output, 'phi_disk_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'scalar')
    io.full_print(omega_disk_output, 'omega_disk_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'vector')
    io.full_print(fsp.sigma_disk_n_12, 'sigma_disk_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'scalar')
    
    # 2.2 on the current mesh
    io.full_print_deformed(phi_disk_output, fsp.u_n_di, 'phi_disk_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, lmsh.sub_meshes[0][0], 'scalar')
    io.full_print_deformed(omega_disk_output, fsp.u_n_di, 'omega_disk_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, lmsh.sub_meshes[0][0], 'vector')
    io.full_print_deformed(fsp.sigma_disk_n_12, fsp.u_n_di, 'sigma_disk_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, lmsh.sub_meshes[0][0], 'scalar')
    
    fi.xdmffile_sigma_di_n_12.write(fsp.sigma_disk_n_12, t)


# print the solution for square fluid 
def print_solution_sq_fluid(t, step):

    # 1 print velocities
    # 1.1 on the reference mesh
    io.full_print(fsp.v_square__, 'v_square__' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][1], 'vector')
    io.full_print(fsp.v_square_n, 'v_square_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][1], 'vector')
    
    # 1.2 on the current mesh
    io.full_print_deformed(fsp.v_square__, fsp.u_n_sq, 'v_square__' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, lmsh.sub_meshes[0][1], 'vector')
    io.full_print_deformed(fsp.v_square_n, fsp.u_n_sq, 'v_square_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, lmsh.sub_meshes[0][1], 'vector')

    fi.xdmffile_v_sq__.write(fsp.v_square__, t)
    fi.xdmffile_v_sq_n.write(fsp.v_square_n, t)


    # 2 print tension 
    # 2.1 on the reference mesh
    io.full_print(fsp.phi_square, 'phi_square_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][1], 'scalar')
    io.full_print(fsp.sigma_square_n_12, 'sigma_square_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][1], 'scalar')
    
    # 2.2 on the current mesh 

    fi.xdmffile_sigma_sq_n_12.write(fsp.sigma_square_n_12, t)



# print the solution for M
def print_solution_M(t, step):

    io.full_print(fsp.c_n, 'c_n_' + str(step), \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
            lmsh.sub_meshes[0][1], 'scalar')

    fi.xdmffile_c_n.write(fsp.c_n, t)


# print solution for all sectors
def print_solution(t, step):

    print_solution_I(t, step)
    print_solution_D(t, step)
    print_solution_di_fluid(t, step)
    print_solution_sq_fluid(t, step)
    print_solution_M(t, step)



# print solution metadata
io.write_parameters_to_csv_file(solpath.csv_files_path + "metadata.csv", rpam.parameters)

