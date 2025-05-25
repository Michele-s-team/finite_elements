'''
This code reads a sequence all .h5 files in a folder, collates them into a time series in xdmf format and writes it into an xdmf file
Run with
    clear; clear; python3 run.py [path of mesh] [path of solution to be read] [path of solution to write]  [increment with which to step from one .h5 file to the next one]

Example:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_IN_PATH="/home/fenics/shared/dynamics/channel_with_cylinder_curved_cn/solution/snapshots/h5"; SOLUTION_OUT_PATH="/home/fenics/shared/read_write/solution"; rm -rf $SOLUTION_OUT_PATH; python3 solve.py square $MESH_PATH $SOLUTION_IN_PATH $SOLUTION_OUT_PATH  2
    MESH_PATH="/home/fenics/shared/generate_mesh/3d/box_ball/solution"; SOLUTION_IN_PATH="/home/fenics/shared/dynamics/channel_with_cylinder_flat_icps/solution/snapshots/h5"; SOLUTION_OUT_PATH="/home/fenics/shared/read_write/solution"; rm -rf $SOLUTION_OUT_PATH; python3 solve.py box_ball $MESH_PATH $SOLUTION_IN_PATH $SOLUTION_OUT_PATH  2
'''

from fenics import *
import importlib
import math
import numpy as np
import meshio
import ufl as ufl
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import load_mesh as lmsh
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

# CHANGE PARAMETERS HERE
# N = (int)(rarg.args.N)
namefile_for_counting = 'u_n_'
N = io.count_files(io.add_trailing_slash(rarg.args.solution_input_directory) + namefile_for_counting, '.h5')
increment = (int)(rarg.args.i)
# CHANGE PARAMETERS HERE

print("Mesh path =", rarg.args.input_directory)
print("Solution in path =", rarg.args.solution_input_directory)
print("Number of snapshots =", N)
print("Solution out path =", rarg.args.output_directory)

# select the appropriate mesh element according to the mesh dimension
# mesh_element = triangle
mesh_element = tetrahedron

# Define function spaces
# finite elements for sigma .... omega
P_v_bar = VectorElement('P', mesh_element, 2)
P_w_bar = FiniteElement('P', mesh_element, 1)
P_phi = FiniteElement('P', mesh_element, 2)
P_v_n = VectorElement('P', mesh_element, 2)
P_w_n = FiniteElement('P', mesh_element, 1)
P_omega_n = VectorElement('P', mesh_element, 3)
P_z_n = FiniteElement('P', mesh_element, 1)

element = MixedElement([P_v_bar, P_w_bar, P_phi, P_v_n, P_w_n, P_omega_n, P_z_n])
Q = FunctionSpace(lmsh.mesh, element)
Q_v_bar = Q.sub(0).collapse()
Q_w_bar = Q.sub(1).collapse()
Q_phi = Q.sub(2).collapse()
Q_v_n = Q.sub(3).collapse()
Q_w_n = Q.sub(4).collapse()
Q_omega_n = Q.sub(5).collapse()
Q_z_n = Q.sub(6).collapse()

XDMF_file_v_n = XDMFFile((rarg.args.output_directory) + '/v_n.xdmf')
XDMF_file_v_bar = XDMFFile((rarg.args.output_directory) + '/v_bar.xdmf')
XDMF_file_w_n = XDMFFile((rarg.args.output_directory) + '/w_n.xdmf')
XDMF_file_w_bar = XDMFFile((rarg.args.output_directory) + '/w_bar.xdmf')
# XDMF_file_phi = XDMFFile( (rarg.args.output_directory) + '/phi.xdmf' )
XDMF_file_sigma_n_12 = XDMFFile((rarg.args.output_directory) + '/sigma_n_12.xdmf')
XDMF_file_omega_n_12 = XDMFFile((rarg.args.output_directory) + '/omega_n_12.xdmf')
XDMF_file_z_n_12 = XDMFFile((rarg.args.output_directory) + '/z_n_12.xdmf')

v_n = Function(Q_v_n)
v_bar = Function(Q_v_bar)
w_n = Function(Q_w_n)
w_bar = Function(Q_w_bar)
sigma_n_12 = Function(Q_phi)
# phi = Function( Q_phi )
omega_n_12 = Function(Q_omega_n)
z_n_12 = Function(Q_z_n)

# Time-stepping
print('Reading snapshots ... ')
for step in range(1, N, increment):
    # time.sleep( 1 )  # Makes Python wait for 5 seconds

    print(f'\tsnapshot # {step}', flush=True)

    # Read the contents of the .h5 files and write them in v, w, .... :
    HDF5File(MPI.comm_world, rarg.args.solution_input_directory + "/u_n_" + str(step) + ".h5", "r").read(v_n, "/f")
    # HDF5File( MPI.comm_world,  rarg.args.solution_input_directory + "/v_bar_" + str(step) + ".h5", "r" ).read(v_bar, "/f" )
    # HDF5File( MPI.comm_world,  rarg.args.solution_input_directory + "/w_n_" + str(step) + ".h5", "r" ).read( w_n, "/f" )
    # HDF5File( MPI.comm_world,  rarg.args.solution_input_directory + "/w_bar_" + str(step) + ".h5", "r" ).read(w_bar, "/f" )
    # HDF5File( MPI.comm_world,  rarg.args.solution_input_directory + "/phi_" + str(step) + ".h5", "r" ).read( phi, "/f" )
    # HDF5File( MPI.comm_world,  rarg.args.solution_input_directory + "/sigma_n_12_" + str(step) + ".h5", "r" ).read( sigma_n_12, "/f" )
    # HDF5File( MPI.comm_world,  rarg.args.solution_input_directory + "/omega_n_12_" + str(step) + ".h5", "r" ).read( omega_n_12, "/f" )
    # HDF5File(MPI.comm_world,  rarg.args.solution_input_directory + "/z_n_12_" + str(step) + ".h5", "r").read(z_n_12, "/f")

    # append into the xdmf files the current time step stored in v, w, ...
    XDMF_file_v_n.write(v_n, step)
    # XDMF_file_v_bar.write( v_bar, step )
    # XDMF_file_w_n.write( w_n, step )
    # XDMF_file_w_bar.write( w_bar, step )
    # XDMF_file_phi.write( phi, step )
    # XDMF_file_sigma_n_12.write( sigma_n_12, step )
    # XDMF_file_omega_n_12.write( omega_n_12, step )
    # XDMF_file_z_n_12.write(z_n_12, step)

    # HDF5_file_write = HDF5File( MPI.comm_world, "solution/snapshots/h5/v_n" + str(step) + ".h5", "w" )
    # HDF5_file_write.write( v, "/f" )
    # HDF5_file_write.close()
print('... done.')
