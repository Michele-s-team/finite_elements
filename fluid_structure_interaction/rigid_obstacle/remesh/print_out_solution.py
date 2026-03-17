'''
this module prints the fields. 
Only some fields are printed. 
'''

from fenics import *

import csv
import importlib
import os
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)


# create the path for the data csv file if it does not exist
data_filename = os.path.join(rarg.args.output_directory, 'data.csv')
os.makedirs(os.path.dirname(data_filename), exist_ok=True)

data_csvfile = open(data_filename, 'a', newline='')
data_fieldnames = [ \
    "theta", \
    "omega", \
    "theta_ref",\
    "mesh_quality"
    ]
data_writer = csv.DictWriter(data_csvfile, fieldnames=data_fieldnames)
data_writer.writeheader()


# create the path for the mesh csv file if it does not exist
remesh_filename = os.path.join(rarg.args.output_directory, 'remesh.csv')
os.makedirs(os.path.dirname(remesh_filename), exist_ok=True)

remesh_csvfile = open(remesh_filename, 'a', newline='')
remesh_fieldnames = [ \
    "step", \
    "mesh_quality"
    ]
remesh_writer = csv.DictWriter(remesh_csvfile, fieldnames=remesh_fieldnames)
remesh_writer.writeheader()


def print_remesh(step, mesh_quality):

    print(f'******** remeshing')

    remesh_writer.writerows([{ \
        remesh_fieldnames[0]: \
            step, 
        remesh_fieldnames[1]: \
            mesh_quality, 
    }])
    remesh_csvfile.flush()
    

def print_solution(step):

    # 1) print theta and omega
    data_writer.writerows([{ \
        data_fieldnames[0]: \
            fsp.theta_n, \
        data_fieldnames[1]: \
            fsp.omega_n,\
        data_fieldnames[2]: \
            rmsh.parameters['phi'],\
        data_fieldnames[3]:
            msh.custom_mesh_quality(msh.deform_mesh(rmsh.lmsh.mesh, fsp.u_n))
    }])
    data_csvfile.flush()


    # 2) print the solution for the mesh problem
    io.full_print(fsp.u_n, 'u_n_' + str(step), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path)
    
    # Write the deformed mesh to file
    deformed_mesh = msh.deform_mesh(lmsh.mesh, fsp.u_n)
    with XDMFFile(solpath.snapshots_path + 'mesh_n_' + str(step) + '.xdmf') as xdmf:
        xdmf.write(deformed_mesh)
    io.print_mesh_vertices_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'vertex_mesh_n_' + str(step) + '.csv')
    io.print_mesh_lines_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'line_mesh_n_' + str(step) + '.csv')


    # 3) print the solution of the fluid problem
    io.full_print(fsp.v_n, 'v_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.sigma_n_12, 'sigma_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    

    io.full_print_deformed(fsp.v_n, fsp.u_n, 'v_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print_deformed(fsp.sigma_n_12, fsp.u_n, 'sigma_n_12_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    


