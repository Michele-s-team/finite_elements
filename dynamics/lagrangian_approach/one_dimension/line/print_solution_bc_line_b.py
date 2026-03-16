from fenics import *
import importlib
import ufl as ufl


import command as cmd
import files as files
import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import solution_paths as solpath
import switch_problem as swi


rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

cmd.set_gauge('arc_length')

i, j, k, l = ufl.indices( 4 )


def print_solution(psi, step, t):

  
    v_bar_dummy, w_bar_dummy, phi_dummy, v_n_dummy, w_n_dummy, u_n_12_dummy, nu_n_12_dummy, psi_n_12_dummy, mu_n_12_dummy = psi.split( deepcopy=True )

 
    fsp.sigma_n_12.assign( fsp.sigma_n_32 - project( phi_dummy, fsp.Q_phi ) )

    # print solution to file
    # append to the full time series solution at the current t
    files.xdmffile_v_bar.write( v_bar_dummy, t )
    files.xdmffile_w_bar.write( w_bar_dummy, t )
    files.xdmffile_v_n.write( v_n_dummy, t )
    files.xdmffile_w_n.write( w_n_dummy, t )
    files.xdmffile_sigma_n_12.write( fsp.sigma_n_12, t - vp.dt / 2.0 )
    files.xdmffile_phi.write( phi_dummy, t )
    files.xdmffile_u_n_12.write( u_n_12_dummy, t - vp.dt / 2.0 )
    files.xdmffile_nu_n_12.write( nu_n_12_dummy, t - vp.dt / 2.0 )
    files.xdmffile_nu_n_12.write( psi_n_12_dummy, t - vp.dt / 2.0 )
    files.xdmffile_mu_n_12.write( mu_n_12_dummy, t - vp.dt / 2.0 )

 
    io.full_print(v_bar_dummy, 'v_bar_' + str(step+1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'vector')
    io.full_print(w_bar_dummy, 'w_bar_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'scalar')
    io.full_print(v_n_dummy, 'v_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'vector')
    io.full_print(w_n_dummy, 'w_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'scalar')
    io.full_print(fsp.sigma_n_12, 'sigma_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'scalar')
    io.full_print(u_n_12_dummy, 'u_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'vector')
    io.full_print(nu_n_12_dummy, 'nu_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'scalar')
    io.full_print(psi_n_12_dummy, 'psi_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'scalar')
    io.full_print(mu_n_12_dummy, 'mu_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'scalar')
    
    
    io.full_print(project(fsp.X_ref + u_n_12_dummy, fsp.Q_X), 'X_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, \
                  'vector')