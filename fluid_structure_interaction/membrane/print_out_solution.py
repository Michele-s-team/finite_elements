from fenics import *


import csv
import function_spaces as fsp
import importlib
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import os
import solution_paths as solpath

import runtime_arguments as rarg
import switch_problem as swi

fi = importlib.import_module(swi.fi)


def print_solution(t, step, dt):
    
    # 1) membrane problem
    v_bar_dummy, w_bar_dummy, phi_dummy, v_n_dummy, w_n_dummy, U_n_12_dummy, nu_n_12_dummy, psi_n_12_dummy, mu_n_12_dummy = fsp.psi_mem.split( deepcopy=True )

    fsp.sigma_n_12.assign( fsp.sigma_n_32 - project( phi_dummy, fsp.Q_phi ) )

    # print solution to file
    # append to the full time series solution at the current t
    fi.xdmffile_v_bar.write( v_bar_dummy, t )
    fi.xdmffile_w_bar.write( w_bar_dummy, t )
    fi.xdmffile_v_n.write( v_n_dummy, t )
    fi.xdmffile_w_n.write( w_n_dummy, t )
    fi.xdmffile_sigma_n_12.write( fsp.sigma_n_12, t - dt / 2.0 )
    fi.xdmffile_phi.write( phi_dummy, t )
    fi.xdmffile_u_n_12.write( U_n_12_dummy, t - dt / 2.0 )
    fi.xdmffile_nu_n_12.write( nu_n_12_dummy, t - dt / 2.0 )
    fi.xdmffile_nu_n_12.write( psi_n_12_dummy, t - dt / 2.0 )
    fi.xdmffile_mu_n_12.write( mu_n_12_dummy, t - dt / 2.0 )

    io.full_print(v_bar_dummy, 'v_bar_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(w_bar_dummy, 'w_bar_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(v_n_dummy, 'v_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(w_n_dummy, 'w_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.sigma_n_12, 'sigma_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(U_n_12_dummy, 'U_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(nu_n_12_dummy, 'nu_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(psi_n_12_dummy, 'psi_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(mu_n_12_dummy, 'mu_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    
    
    io.full_print(project(fsp.X_ref + U_n_12_dummy, fsp.Q_X), 'X_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)


    # 2) mesh problem
    io.full_print(fsp.u_n, 'u_n_' + str(step + 1), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path,
                  solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.u_dot_n, 'u_dot_n_' + str(step + 1), solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

    # include the snapshot in xdmf files
    fi.xdmffile_u_n.write(fsp.u_n, t)
    fi.xdmffile_u_dot_n.write(fsp.u_dot_n, t)

    # Write the deformed mesh to file
    deformed_mesh = msh.deform_mesh(lmsh.sub_meshes[0], fsp.u_n)
    with XDMFFile(solpath.snapshots_path + 'mesh_n_' + str(step + 1) + '.xdmf') as xdmf:
        xdmf.write(deformed_mesh)
    io.print_mesh_vertices_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'vertex_mesh_n_' + str(step + 1) + '.csv')
    io.print_mesh_lines_to_csv(deformed_mesh, solpath.snapshots_csv_path + 'line_mesh_n_' + str(step + 1) + '.csv')



    # 3) fluid problem
    io.full_print(fsp.v_fl_bar, 'v_fl_bar_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.v_fl_n, 'v_fl_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.sigma_fl_n_12, 'sigma_fl_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.phi_fl, 'phi_fl_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

    # include the snapshot in xdmf files
    fi.xdmffile_v_fl_n.write(fsp.v_fl_n, t)
    fi.xdmffile_v_fl_bar.write(fsp.v_fl_bar, t)
    fi.xdmffile_sigma_fl.write(fsp.sigma_fl_n_12, t - dt / 2.0)
    fi.xdmffile_phi_fl.write(fsp.phi_fl, t)

    io.full_print_deformed(fsp.v_fl_bar, fsp.u_n, 'v_fl_bar_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print_deformed(fsp.v_fl_n, fsp.u_n, 'v_fl_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print_deformed(fsp.sigma_fl_n_12, fsp.u_n, 'sigma_fl_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print_deformed(fsp.phi_fl, fsp.u_n, 'phi_fl_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)



