from fenics import *
import importlib
import os

import differential_geometry.manifold.geometry as geo
import files as fi
import input_output as io
import mesh.utils as msh
import physics.elasticity as ela
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)


def print_solution(t, step, dt):

    #1 unpack the mixed field 

    v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy = fsp.psi.split( deepcopy=True )
    det_F_n = project(ela.detF(u_n_dummy), fsp.Q_det_F)

    #2 write to xdmf files

    fi.xdmffile_v_n.write(v_n_dummy, t)
    fi.xdmffile_sigma_n.write(sigma_n_dummy, t)

    fi.xdmffile_u_n.write(u_n_dummy, t)
    fi.xdmffile_u_dot_n.write(u_dot_n_dummy, t)

    fi.xdmffile_det_F_n.write(det_F_n, t)



    # 3 write snapshots

    # 3.1 reference configuration and deformation fields
    io.full_print(v_n_dummy, 'v_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])    
    io.full_print(sigma_n_dummy, 'sigma_n_' + str(step), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])

    
    io.full_print(u_n_dummy, 'u_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
    io.full_print(u_dot_n_dummy, 'u_dot_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])

    # 3.2 current configuration
    io.full_print_deformed(v_n_dummy, u_n_dummy, 'v_n_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
    
    io.full_print_deformed(sigma_n_dummy, u_n_dummy, 'sigma_n_' + str(step), \
                           solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
    
    # 3.3 average of det(F)
    io.full_print(det_F_n, 'det_F_n_' + str(step), \
                solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])



    #4. Write the deformed mesh to file

    # 4.1 write mesh vertices and lines
    deformed_mesh = msh.deform_mesh(rmsh.lmsh.mesh[0], u_n_dummy)

    with XDMFFile(solpath.snapshots_path + 'mesh_n_' + str(step) + '.xdmf') as xdmf:
        xdmf.write(deformed_mesh)

    io.print_mesh_vertices_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'vertex_mesh_n_' + str(step) + '.csv')
    io.print_mesh_lines_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'line_mesh_n_' + str(step) + '.csv')

    # 4.2 write shape vertices 
    input_path = os.path.join(rarg.args.input_directory, f"mesh_0/boundary_points_id_{rmsh.parameters['shape_id']}.csv")
    output_path = os.path.join(rarg.args.output_directory, f"snapshots/csv/boundary_points_id_{rmsh.parameters['shape_id']}_n_{step}.csv")
    os.system(f'cp {input_path} {output_path}')






