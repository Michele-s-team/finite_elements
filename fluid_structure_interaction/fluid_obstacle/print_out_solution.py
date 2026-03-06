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
def print_solution_I(step):

    io.full_print(fsp.U_n_12, 'U_n_12_' + str(step), \
            solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
            lmsh.mesh[1], 'vector')


# print the solution for D 
def print_solution_D(step):
   
    #1 print fields for di    
    io.full_print(fsp.u_n_di, 'u_di_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'vector')
    
    io.full_print(fsp.u_n_di_dot, 'u_di_dot_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'vector')
    
    
    # print fields for sq
    io.full_print(fsp.u_n_sq, 'u_sq_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][1], 'vector')
    
    io.full_print(fsp.u_n_sq_dot, 'u_sq_dot_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][1], 'vector')
    


# print the solution for disk fluid 
def print_solution_di_fluid(t, step):

    phi_disk_output, omega_disk_output = fsp.phi_omega_disk.split(deepcopy=True)

    # 1 print velocities
    io.full_print(fsp.v_disk__, 'v_disk__' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'vector')
    io.full_print(fsp.v_disk_n, 'v_disk_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'vector')

    # 2 print tension 
    io.full_print(phi_disk_output, 'phi_disk_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'scalar')
    io.full_print(omega_disk_output, 'omega_disk_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][0], 'vector')

# print the solution for square fluid 
def print_solution_sq_fluid(t, step):

    # 1 print velocities
    io.full_print(fsp.v_square__, 'v_square__' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  lmsh.sub_meshes[0][1], 'vector')


# print solution for all sectors
def print_solution(t, step):

    print_solution_I(step)
    print_solution_D(step)
    print_solution_di_fluid(t, step)
    print_solution_sq_fluid(t, step)



# print solution metadata
io.write_parameters_to_csv_file(solpath.csv_files_path + "metadata.csv", rpam.parameters)

