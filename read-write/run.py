'''
this code reads a sequence of .h5 files, collates them into a time series in xdmf format and writes it into an xdmf file
run with
clear; clear; python3 run.py [path of mesh] [path of solution to be read] [path of solution to write]  [number of .h5 files to be read] [increment with which to step from one .h5 file to the next one]
clear; clear; rm -rf solution-out; python3 run.py /home/fenics/shared/dynamics/mesh /home/fenics/shared/dynamics/solution/snapshots/h5  /home/fenics/shared/read-write/solution-out 2673 10
'''

from fenics import *
import math
from mshr import *
import numpy as np
import meshio
import ufl as ufl
import argparse

import sys

#add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import geometry as geo

parser = argparse.ArgumentParser()
parser.add_argument("mesh_old_directory")
parser.add_argument("solution_old_directory")
parser.add_argument("solution_new_directory")
parser.add_argument("N")
parser.add_argument("i")
args = parser.parse_args()

print("mesh old folder =", args.mesh_old_directory)
print("solution old folder =", args.solution_old_directory)
print("solution new folder =", args.solution_new_directory)

#CHANGE PARAMETERS HERE
L = 1.0
h = L
r = 0.125
c_r = [L/2.0, h/2.0]
N = (int)(args.N)
increment = (int)(args.i)
# time step size
#CHANGE PARAMETERS HERE


#read mesh
mesh=Mesh()
with XDMFFile((args.mesh_old_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.mesh_old_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")


#this is the facet normal vector, which cannot be plotted as a field
#n_overline = \overline{n}_notes_{on the circle}
n_overline = FacetNormal(mesh)

# Define function spaces
#finite elements for sigma .... omega
P_v_bar = VectorElement( 'P', triangle, 2 )
P_w_bar = FiniteElement( 'P', triangle, 1 )
P_phi = FiniteElement('P', triangle, 2)
P_v_n = VectorElement( 'P', triangle, 2 )
P_w_n = FiniteElement( 'P', triangle, 1 )
P_omega_n = VectorElement( 'P', triangle, 3 )
P_z_n = FiniteElement( 'P', triangle, 1 )

element = MixedElement( [P_v_bar, P_w_bar, P_phi, P_v_n, P_w_n, P_omega_n, P_z_n] )
#total function space
Q = FunctionSpace(mesh, element)
#function spaces for vbar .... zn
Q_v_bar = Q.sub(0).collapse()
Q_w_bar = Q.sub(1).collapse()
Q_phi = Q.sub(2).collapse()
Q_v_n = Q.sub(3).collapse()
Q_w_n = Q.sub(4).collapse()
Q_omega_n = Q.sub(5).collapse()
Q_z_n= Q.sub(6).collapse()



# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_l  = 'near(x[0], 0.0)'
boundary_r  = 'near(x[0], 1.0)'
boundary_lr  = 'near(x[0], 0) || near(x[0], 1.0)'
boundary_tb  = 'near(x[1], 0) || near(x[1], 1.0)'
boundary_square = 'on_boundary && sqrt(pow(x[0] - 1.0/2.0, 2) + pow(x[1] - 1.0/2.0, 2)) > (0.125 + 1.0/2.0)/2.0'
boundary_circle = 'on_boundary && sqrt(pow(x[0] - 1.0/2.0, 2) + pow(x[1] - 1.0/2.0, 2)) < (0.125 + 1.0/2.0)/2.0'
#CHANGE PARAMETERS HERE


XDMF_file_v_n = XDMFFile( (args.solution_new_directory) + '/v_n.xdmf' )
XDMF_file_v_bar = XDMFFile( (args.solution_new_directory) + '/v_bar.xdmf' )
XDMF_file_w_n = XDMFFile( (args.solution_new_directory) + '/w_n.xdmf' )
XDMF_file_w_bar = XDMFFile( (args.solution_new_directory) + '/w_bar.xdmf' )
# XDMF_file_phi = XDMFFile( (args.solution_new_directory) + '/phi.xdmf' )
XDMF_file_sigma_n_12 = XDMFFile( (args.solution_new_directory) + '/sigma_n_12.xdmf' )
XDMF_file_omega_n_12 = XDMFFile( (args.solution_new_directory) + '/omega_n_12.xdmf' )
XDMF_file_z_n_12 = XDMFFile( (args.solution_new_directory) + '/z_n_12.xdmf' )

v_n = Function( Q_v_n )
v_bar = Function(Q_v_bar)
w_n = Function( Q_w_n )
w_bar = Function(Q_w_bar)
sigma_n_12 = Function( Q_phi )
# phi = Function( Q_phi )
omega_n_12 = Function( Q_omega_n )
z_n_12 = Function( Q_z_n )

# Time-stepping
for step in range(1, N, increment):
    # time.sleep( 1 )  # Makes Python wait for 5 seconds

    print("* step = ", step, "\n")

    # Read the contents of the .h5 files and write them in v, w, .... :
    # HDF5File( MPI.comm_world, (args.solution_old_directory) + "/v_n_" + str(step) + ".h5", "r" ).read( v_n, "/f" )
    # HDF5File( MPI.comm_world, (args.solution_old_directory) + "/v_bar_" + str(step) + ".h5", "r" ).read(v_bar, "/f" )
    # HDF5File( MPI.comm_world, (args.solution_old_directory) + "/w_n_" + str(step) + ".h5", "r" ).read( w_n, "/f" )
    # HDF5File( MPI.comm_world, (args.solution_old_directory) + "/w_bar_" + str(step) + ".h5", "r" ).read(w_bar, "/f" )
    # HDF5File( MPI.comm_world, (args.solution_old_directory) + "/phi_" + str(step) + ".h5", "r" ).read( phi, "/f" )
    # HDF5File( MPI.comm_world, (args.solution_old_directory) + "/sigma_n_12_" + str(step) + ".h5", "r" ).read( sigma_n_12, "/f" )
    # HDF5File( MPI.comm_world, (args.solution_old_directory) + "/omega_n_12_" + str(step) + ".h5", "r" ).read( omega_n_12, "/f" )
    HDF5File( MPI.comm_world, (args.solution_old_directory) + "/z_n_12_" + str(step) + ".h5", "r" ).read( z_n_12, "/f" )

    # append into the xdmf files the current time step stored in v, w, ...
    # XDMF_file_v_n.write( v_n, step )
    # XDMF_file_v_bar.write( v_bar, step )
    # XDMF_file_w_n.write( w_n, step )
    # XDMF_file_w_bar.write( w_bar, step )
    # XDMF_file_phi.write( phi, step )
    # XDMF_file_sigma_n_12.write( sigma_n_12, step )
    # XDMF_file_omega_n_12.write( omega_n_12, step )
    XDMF_file_z_n_12.write( z_n_12, step )

    # HDF5_file_write = HDF5File( MPI.comm_world, "solution/snapshots/h5/v_n" + str(step) + ".h5", "w" )
    # HDF5_file_write.write( v, "/f" )
    # HDF5_file_write.close()