import csv
from fenics import *
import importlib
import os
import ufl as ufl


import command as cmd
import differential_geometry.boundary.geometry as bgeo
import files as files
import function_spaces as fsp
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi


rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

cmd.set_gauge('arc_length')

i, j, k, l = ufl.indices( 4 )

#set up printout of the BCs to file
# create the path for the csv file if it does not exist
os.makedirs(os.path.dirname(rarg.args.output_directory + '/bcs.csv'), exist_ok=True)

csvfile_bcs = open( (rarg.args.output_directory) + '/bcs.csv', 'a', newline='' )
fieldnames_bcs = [ \
    '<<|v_bar - v_bar_l|^2>>_{partial Omega l}', \
    '<<|v_bar - v_bar_r|^2>>_{partial Omega r}', \
    '<<(w_bar)^2>>_{partial Omega l}', \
    '<<(Nabla_i w_bar)^2>>_{partial Omega r}', \
    '<<(n^i Nabla_i phi)^2>>_{partial Omega}', \
    '<<|u_n_12 - u_n_12_l|^2>>_{partial Omega l}', \
    '<<u_n_12_0^2>>_{partial Omega r}', \
    '<<(Nabla_i u_n_12_1)^2>>_{partial Omega r}', \
    ]
writer_bcs = csv.DictWriter( csvfile_bcs, fieldnames=fieldnames_bcs )
writer_bcs.writeheader()



# this function prints out the residuals of BCs
def print_bcs(psi):
    # get the solution and write it to file

    v_bar_dummy, w_bar_dummy, phi_dummy, v_n_dummy, w_n_dummy, u_n_12_dummy, nu_n_12_dummy, psi_n_12_dummy, mu_n_12_dummy = psi.split( deepcopy=True )

    # write the residual of natural BCs on step 2 to file
    writer_bcs.writerows( [{ \
        fieldnames_bcs[0]: \
            f"{msh.abs_wrt_measure( sqrt((v_bar_dummy[0] - rpam.parameters['v_bar_l'][0]) * (v_bar_dummy[0] - rpam.parameters['v_bar_l'][0])), rmsh.ds_l ):.{io.number_of_decimals}e}",\
        fieldnames_bcs[1]: \
            f"{msh.abs_wrt_measure( sqrt((v_bar_dummy[0] - rpam.parameters['v_bar_r'][0]) * (v_bar_dummy[0] - rpam.parameters['v_bar_r'][0])), rmsh.ds_r ):.{io.number_of_decimals}e}",\
        fieldnames_bcs[2]: \
            f"{msh.abs_wrt_measure( w_bar_dummy, rmsh.ds_l ):.{io.number_of_decimals}e}",\
        fieldnames_bcs[3]: \
            f"{msh.abs_wrt_measure( (w_bar_dummy.dx( 0 )), rmsh.ds_r ):.{io.number_of_decimals}e}",\
        fieldnames_bcs[4]: \
            f"{msh.abs_wrt_measure( (bgeo.n_lr( psi_n_12_dummy, nu_n_12_dummy ))[i] * (fsp.phi.dx( i )), rmsh.ds ):.{io.number_of_decimals}e}",\
        fieldnames_bcs[5]: \
            f"{msh.abs_wrt_measure(sqrt((u_n_12_dummy[0] - rpam.parameters['u_n_12_l'][0])**2 + (u_n_12_dummy[1] - rpam.parameters['u_n_12_l'][1])**2), rmsh.ds_l):.{io.number_of_decimals}e}",\
        fieldnames_bcs[6]: \
            f"{msh.abs_wrt_measure(u_n_12_dummy[0], rmsh.ds_r):.{io.number_of_decimals}e}",\
        fieldnames_bcs[7]: \
            f"{msh.abs_wrt_measure((u_n_12_dummy[1]).dx(0), rmsh.ds_r):.{io.number_of_decimals}e}",\
        }] )
    csvfile_bcs.flush()


def print_solution(psi, step, t):

    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, u_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = psi.split( deepcopy=True )
    
    
    fsp.sigma_n_12_output.assign( fsp.sigma_n_32 - project( phi_output, fsp.Q_phi ) )

    # print solution to file
    # append to the full time series solution at the current t
    files.xdmffile_v_bar.write( v_bar_output, t )
    files.xdmffile_w_bar.write( w_bar_output, t )
    files.xdmffile_phi.write( phi_output, t )
    files.xdmffile_v_n.write( v_n_output, t )
    files.xdmffile_w_n.write( w_n_output, t )
    files.xdmffile_sigma_n_12.write( fsp.sigma_n_12_output, t - vp.dt / 2.0 )
    files.xdmffile_phi.write( phi_output, t )
    files.xdmffile_u_n_12.write( u_n_12_output, t - vp.dt / 2.0 )
    files.xdmffile_nu_n_12.write( nu_n_12_output, t - vp.dt / 2.0 )
    files.xdmffile_psi_n_12.write( psi_n_12_output, t - vp.dt / 2.0 )
    files.xdmffile_mu_n_12.write( mu_n_12_output, t - vp.dt / 2.0 )

 
    io.full_print(v_bar_output, 'v_bar_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(w_bar_output, 'w_bar_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(v_n_output, 'v_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(w_n_output, 'w_n_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.sigma_n_12_output, 'sigma_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(fsp.phi, 'phi_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(u_n_12_output, 'u_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(nu_n_12_output, 'nu_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(psi_n_12_output, 'psi_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
    io.full_print(mu_n_12_output, 'mu_n_12_' + str(step + 1), \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)