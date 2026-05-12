import csv
from fenics import *
import importlib

import differential_geometry.manifold.geometry as geo
import files as fi
import function_spaces as fsp
import input_output as io
import mesh.utils as msh
import os
import parameters.read.solution as rpam
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi


rmsh = importlib.import_module(swi.rmsh)




def print_solution(t, step, dt):

    #1 unpack the mixed field 

    v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy = fsp.psi.split( deepcopy=True )


    #2 write to xdmf files

    fi.xdmffile_v_n.write(v_n_dummy, t)
    fi.xdmffile_sigma_n.write(sigma_n_dummy, t)

    fi.xdmffile_u_n.write(u_n_dummy, t)
    fi.xdmffile_u_dot_n.write(u_dot_n_dummy, t)


    # 3 write snapshots

    # 3.1 reference configuration and deformation fields
    io.full_print(v_n_dummy, 'v_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)    
    io.full_print(sigma_n_dummy, 'sigma_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)

    
    io.full_print(u_n_dummy, 'u_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)
    io.full_print(u_dot_n_dummy, 'u_dot_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)

    # 3.2 current configuration
    io.full_print_deformed(v_n_dummy, u_n_dummy, 'v_n_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)
    
    io.full_print_deformed(sigma_n_dummy, u_n_dummy, 'sigma_n_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)



    #4. Write the deformed mesh to file
    deformed_mesh = msh.deform_mesh(rmsh.lmsh.mesh, u_n_dummy)

    with XDMFFile(solpath.snapshots_path + 'mesh_n_' + str(step) + '.xdmf') as xdmf:
        xdmf.write(deformed_mesh)

    io.print_mesh_vertices_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'vertex_mesh_n_' + str(step) + '.csv')
    io.print_mesh_lines_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'line_mesh_n_' + str(step) + '.csv')






