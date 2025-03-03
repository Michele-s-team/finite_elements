from fenics import *
from mshr import *
import ufl as ufl

import csv

import boundary_geometry as bgeo
import files as files
import function_spaces as fsp
import geometry as geo
import input_output as io
import mesh as msh
import physics as phys
import read_mesh as rmsh
import runtime_arguments as rarg
import variational_problem_bc_b as vp

i, j, k, l = ufl.indices( 4 )

#set up printout of the BCs to file
csvfile_bcs = open( (rarg.args.output_directory) + '/bcs.csv', 'a', newline='' )
fieldnames_bcs = [ \
    '<<(n^{n-1/2}_i d^{i 1})^2>>_R', \
    '<<(n^i Nabla_i phi)^2>>_{L + W + O}', \
    '<<(n^{n-1/2}_i \overline{v}^i)^2>>_{W + O}', \
    '<<(n^{n-1/2}^i \omega^{n-1/2}_i)^2 >>', \
    '<<(l_profile_v_bar^i - v_bar^i)(l_profile_v_bar_i - v_bar_i)>>_L', \
    '<<(w_bar - boundary_profile_w_bar)^2>>', \
    '<<(phi - r_profile_phi)^2>>', \
    '<<(z - boundary_profile_z)^2>>'
]
writer_bcs = csv.DictWriter( csvfile_bcs, fieldnames=fieldnames_bcs )
writer_bcs.writeheader()

#set up printout to file of the force F on the obstacle
csvfile_F = open( (rarg.args.output_directory) + '/F.csv', 'a', newline='' )
fieldnames_F = [ 'F_circle^1', 'F_circle^2']
writer_F = csv.DictWriter( csvfile_F, fieldnames=fieldnames_F )
writer_F.writeheader()


# this function prints out the residuals of BCs
def print_bcs(psi):
    # get the solution and write it to file

    v_bar_dummy, w_bar_dummy, phi_dummy, v_n_dummy, w_n_dummy, z_n_12_dummy, omega_n_12_dummy, mu_n_12_dummy = psi.split( deepcopy=True )

    # write the residual of natural BCs on step 2 to file
    writer_bcs.writerows( [{ \
        fieldnames_bcs[0]: \
            f"{msh.abs_wrt_measure(geo.d_c( (v_bar_dummy + fsp.v_n_1) / 2.0, (w_bar_dummy + fsp.w_n_1) / 2.0, omega_n_12_dummy )[i, 0] * geo.g( omega_n_12_dummy )[i, k] * (bgeo.n_lr( omega_n_12_dummy ))[k], rmsh.ds_r ):.{io.number_of_decimals}e}", \
        fieldnames_bcs[1]: \
            f"{msh.abs_wrt_measure( (bgeo.n_lr( omega_n_12_dummy ))[i] * (phi_dummy.dx( i )), rmsh.ds_l ) + msh.abs_wrt_measure( (bgeo.n_tb( omega_n_12_dummy ))[i] * (phi_dummy.dx( i )), rmsh.ds_tb ) + msh.abs_wrt_measure( (bgeo.n_circle( omega_n_12_dummy ))[i] * (phi_dummy.dx( i )), rmsh.ds_circle ):.{io.number_of_decimals}e}", \
        fieldnames_bcs[2]: \
            f"{msh.abs_wrt_measure( v_bar_dummy[i] * geo.g( omega_n_12_dummy )[i, j] * (bgeo.n_tb( omega_n_12_dummy ))[j], rmsh.ds_tb ) + msh.abs_wrt_measure( v_bar_dummy[i] * geo.g( omega_n_12_dummy )[i, j] * (bgeo.n_circle( omega_n_12_dummy ))[j], rmsh.ds_circle ):.{io.number_of_decimals}e}", \
        fieldnames_bcs[3]: \
            f"{msh.abs_wrt_measure( (bgeo.n_lr( omega_n_12_dummy ))[i] * omega_n_12_dummy[i] - vp.omega_n_square, rmsh.ds_lr ) + msh.abs_wrt_measure( (bgeo.n_tb( omega_n_12_dummy ))[i] * omega_n_12_dummy[i] - vp.omega_n_square, rmsh.ds_tb ) + msh.abs_wrt_measure( (bgeo.n_circle( omega_n_12_dummy ))[i] * omega_n_12_dummy[i] - vp.omega_n_circle, rmsh.ds_circle ):.{io.number_of_decimals}e}", \
        fieldnames_bcs[4]: \
            f"{msh.abs_wrt_measure( sqrt((vp.l_profile_v_bar[i] - v_bar_dummy[i]) * (vp.l_profile_v_bar[i] - v_bar_dummy[i])), rmsh.ds_l ):.{io.number_of_decimals}e}", \
        fieldnames_bcs[5]: \
            f"{msh.abs_wrt_measure( w_bar_dummy - vp.boundary_profile_w_bar, rmsh.ds ):.{io.number_of_decimals}e}", \
        fieldnames_bcs[6]: \
            f"{msh.abs_wrt_measure( phi_dummy - vp.r_profile_phi, rmsh.ds_r ):.{io.number_of_decimals}e}", \
        fieldnames_bcs[7]: \
            f"{msh.abs_wrt_measure( z_n_12_dummy - vp.boundary_profile_z, rmsh.ds ):.{io.number_of_decimals}e}", \
        }] )
    csvfile_bcs.flush()


def print_solution(psi, step, t):

    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, z_n_12_output, omega_n_12_output, mu_n_12_output = fsp.psi.split( deepcopy=True )
    fsp.sigma_n_12_output.assign( fsp.sigma_n_32 - project( phi_output, fsp.Q_phi ) )

    # print solution to file
    # append to the full time series solution at the current t
    files.xdmffile_v_bar.write( v_bar_output, t )
    files.xdmffile_w_bar.write( w_bar_output, t )
    files.xdmffile_v.write( v_n_output, t )
    files.xdmffile_w.write( w_n_output, t )
    files.xdmffile_sigma.write( fsp.sigma_n_12_output, t - vp.dt / 2.0 )
    files.xdmffile_phi.write( phi_output, t )
    files.xdmffile_z.write( z_n_12_output, t - vp.dt / 2.0 )
    files.xdmffile_omega.write( omega_n_12_output, t - vp.dt / 2.0 )
    files.xdmffile_mu.write( mu_n_12_output, t - vp.dt / 2.0 )

    files.xdmffile_nu.write( fsp.nu_n_12, t - vp.dt / 2.0 )
    files.xdmffile_tau.write( fsp.tau_n_12, t - vp.dt / 2.0 )
    files.xdmffile_d.write( fsp.d, t )

    # write the solution at current step, so, in case the code crashes, it can be read back
    # write the solutions in .h5 format into  snapshots/h5
    HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/snapshots/h5/v_bar_" + str( step + 1 ) + ".h5", "w" ).write( v_bar_output, "/f" )
    HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/snapshots/h5/w_bar_" + str( step + 1 ) + ".h5", "w" ).write( w_bar_output, "/f" )
    HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/snapshots/h5/v_n_" + str( step + 1 ) + ".h5", "w" ).write( v_n_output, "/f" )
    HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/snapshots/h5/w_n_" + str( step + 1 ) + ".h5", "w" ).write( w_n_output, "/f" )
    HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/snapshots/h5/sigma_n_12_" + str( step + 1 ) + ".h5", "w" ).write( fsp.sigma_n_12_output, "/f" )
    HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/snapshots/h5/z_n_12_" + str( step + 1 ) + ".h5", "w" ).write( z_n_12_output, "/f" )
    HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/snapshots/h5/omega_n_12_" + str( step + 1 ) + ".h5", "w" ).write( omega_n_12_output, "/f" )
    HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/snapshots/h5/mu_n_12_" + str( step + 1 ) + ".h5", "w" ).write( mu_n_12_output, "/f" )

    HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/snapshots/h5/nu_n_12_" + str( step + 1 ) + ".h5", "w" ).write( fsp.nu_n_12, "/f" )
    HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/snapshots/h5/tau_n_12_" + str( step + 1 ) + ".h5", "w" ).write( fsp.tau_n_12, "/f" )
    HDF5File( MPI.comm_world, (rarg.args.output_directory) + "/snapshots/h5/d_n" + str( step + 1 ) + ".h5", "w" ).write( fsp.d, "/f" )

    # write the solutions in .xdmf format into  snapshots/xdmf
    XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/v_bar_' + str( step + 1 ) + '.xdmf' ).write( v_bar_output )
    XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/w_bar_' + str( step + 1 ) + '.xdmf' ).write( w_bar_output )
    XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/v_n_' + str( step + 1 ) + '.xdmf' ).write( v_n_output )
    XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/w_n_' + str( step + 1 ) + '.xdmf' ).write( w_n_output )
    XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/sigma_n_12_' + str( step + 1 ) + '.xdmf' ).write( fsp.sigma_n_12_output )
    XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/z_n_12_' + str( step + 1 ) + '.xdmf' ).write( z_n_12_output )
    XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/omega_n_12_' + str( step + 1 ) + '.xdmf' ).write( omega_n_12_output )
    XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/mu_n_12_' + str( step + 1 ) + '.xdmf' ).write( mu_n_12_output )

    XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/nu_n_12_' + str( step + 1 ) + '.xdmf' ).write( fsp.nu_n_12 )
    XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/tau_n_12_' + str( step + 1 ) + '.xdmf' ).write( fsp.tau_n_12 )
    XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/d_n' + str( step + 1 ) + '.xdmf' ).write( fsp.d )

    io.print_vector_to_csvfile( v_bar_output, (rarg.args.output_directory) + '/snapshots/csv/v_bar_' + str( step + 1 ) + '.csv' )
    io.print_scalar_to_csvfile( w_bar_output, (rarg.args.output_directory) + '/snapshots/csv/w_bar_' + str( step + 1 ) + '.csv' )
    io.print_vector_to_csvfile( v_n_output, (rarg.args.output_directory) + '/snapshots/csv/v_n_' + str( step + 1 ) + '.csv' )
    io.print_scalar_to_csvfile( w_n_output, (rarg.args.output_directory) + '/snapshots/csv/w_n_' + str( step + 1 ) + '.csv' )
    io.print_scalar_to_csvfile( fsp.sigma_n_12_output, (rarg.args.output_directory) + '/snapshots/csv/sigma_n_12_' + str( step + 1 ) + '.csv' )
    io.print_scalar_to_csvfile( z_n_12_output, (rarg.args.output_directory) + '/snapshots/csv/z_n_12_' + str( step + 1 ) + '.csv' )
    io.print_vector_to_csvfile( omega_n_12_output, (rarg.args.output_directory) + '/snapshots/csv/omega_n_12_' + str( step + 1 ) + '.csv' )
    io.print_scalar_to_csvfile( mu_n_12_output, (rarg.args.output_directory) + '/snapshots/csv/mu_n_12_' + str( step + 1 ) + '.csv' )

    io.print_vector_to_csvfile( fsp.nu_n_12, (rarg.args.output_directory) + '/snapshots/csv/nu_' + str( step + 1 ) + '.csv' )
    io.print_scalar_to_csvfile( fsp.tau_n_12, (rarg.args.output_directory) + '/snapshots/csv/tau_' + str( step + 1 ) + '.csv' )


    # print to file the forces which appear in the RHS of the equations (5a) in notes
    # tangential forces
    fsp.f_visc_t.assign( project( phys.fvisc_t( fsp.d, omega_n_12_output, vp.eta ), fsp.Q_f_t ) )
    fsp.f_sigma_t.assign(project( phys.fsigma_t( fsp.sigma_n_32, omega_n_12_output ), fsp.Q_f_t ))
    fsp.f_v_t.assign( project( phys.ma_cn_t( v_bar_output, fsp.v_n_1, fsp.v_n_2, w_bar_output, fsp.w_n_1, omega_n_12_output, vp.rho, vp.dt ), fsp.Q_f_t ) )

    files.xdmffile_f.write( fsp.f_visc_t, t )
    files.xdmffile_f.write( fsp.f_sigma_t, t )
    files.xdmffile_f.write( fsp.f_v_t, t )

    io.print_vector_to_csvfile( fsp.f_visc_t, (rarg.args.output_directory)  + '/snapshots/csv/fvisc_t_' + str( step + 1 )  + '.csv' )
    io.print_vector_to_csvfile( fsp.f_sigma_t, (rarg.args.output_directory)  + '/snapshots/csv/fsigma_t_' + str( step + 1 )  + '.csv' )
    io.print_vector_to_csvfile( fsp.f_v_t, (rarg.args.output_directory)  + '/snapshots/csv/fv_t_' + str( step + 1 )  + '.csv' )


    # normal forces
    fsp.f_visc_n.assign(project( phys.fvisc_n( fsp.V, fsp.W, omega_n_12_output, fsp.mu_n_12, vp.eta ), fsp.Q_f_n ))
    fsp.f_el_n.assign(project( phys.fel_n( omega_n_12_output, mu_n_12_output, fsp.tau_n_12, vp.kappa ), fsp.Q_f_n ))
    fsp.f_laplace.assign( project( phys.flaplace( fsp.sigma_n_32, omega_n_12_output ), fsp.Q_f_n ) )

    files.xdmffile_f.write( fsp.f_visc_n, t )
    files.xdmffile_f.write( fsp.f_el_n, t )
    files.xdmffile_f.write( fsp.f_laplace, t )

    io.print_scalar_to_csvfile( fsp.f_visc_n, (rarg.args.output_directory) + '/snapshots/csv/fvisc_n_' + str( step + 1 )  + '.csv')
    io.print_scalar_to_csvfile( fsp.f_el_n, (rarg.args.output_directory) + '/snapshots/csv/fel_n_' + str( step + 1 )  + '.csv' )
    io.print_scalar_to_csvfile( fsp.f_laplace, (rarg.args.output_directory) + '/snapshots/csv/flaplace_' + str( step + 1 )  + '.csv' )

    # print rate of deformation tensor to file
    # files.xdmffile_d.write( project( fsp.d, fsp.Q_d ), t )



    #print tangential (normal) force per unit length (surface)

    fsp.dFdl.assign( project( phys.dFdl( v_n_output, w_n_output, omega_n_12_output, fsp.sigma_n_12, vp.eta, geo.n_c_r( bgeo.mesh, rmsh.c_r, omega_n_12_output ) ), fsp.Q_dFdl ) )
    fsp.dFds.assign( project( phys.ma_cn_n(v_bar_output, fsp.v_n_1, fsp.v_n_2, w_bar_output, fsp.w_n_1, omega_n_12_output, vp.rho, vp.dt), fsp.Q_dFds ) )

    files.xdmffile_dFdl.write( fsp.dFdl, t )
    files.xdmffile_dFds.write( fsp.dFds, t )

    io.print_vector_to_csvfile( fsp.dFdl, (rarg.args.output_directory) + '/snapshots/csv/dFdl_' + str( step + 1 )  + '.csv' )
    io.print_vector_to_csvfile( fsp.dFds, (rarg.args.output_directory) + '/snapshots/csv/dFds_' + str( step + 1 )  + '.csv' )


    xdmffile_dFdlds = XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/dFdlds_' + str( step + 1 ) + '.xdmf' )
    xdmffile_dFdlds.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

    xdmffile_dFdlds.write( fsp.dFdl, 0 )
    xdmffile_dFdlds.write( fsp.dFds, 0 )

    xdmffile_dFdlds.close()

    

    #print residuals of variational problems
    fsp.res_F_omega_n.assign( project( sqrt( ((z_n_12_output.dx( i )) - omega_n_12_output[i]) * ((z_n_12_output.dx( i )) - omega_n_12_output[i]) ), fsp.Q_f_n ) )
    # fsp.res_F_v_bar.assign( project( fsp.f_v_t - fsp.f_sigma_t - fsp.f_visc_t, fsp.Q_v_bar ) )


    # print residual of the PDEs to files
    xdmffile_check = XDMFFile( (rarg.args.output_directory) + '/snapshots/xdmf/check_' + str( step + 1 ) + '.xdmf'  )
    xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

    xdmffile_check.write( fsp.res_F_omega_n, 0 )
    # xdmffile_check.write( project(sqrt(fsp.res_F_v_bar[i] * fsp.res_F_v_bar[i]), fsp.Q_z_n), 0 )

    xdmffile_check.close()

    # write the force F extered on ds_circle to file
    writer_F.writerows( [{ \
        fieldnames_F[0]: \
            f"{assemble( phys.dFdl( v_n_output, w_n_output, omega_n_12_output, fsp.sigma_n_12_output, vp.eta, geo.n_c_r( bgeo.mesh, rmsh.c_r, omega_n_12_output ) )[0] * bgeo.sqrt_deth_circle( omega_n_12_output, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle )}", \
        fieldnames_F[1]: \
            f"{assemble( phys.dFdl( v_n_output, w_n_output, omega_n_12_output, fsp.sigma_n_12_output, vp.eta, geo.n_c_r( bgeo.mesh, rmsh.c_r, omega_n_12_output ) )[1] * bgeo.sqrt_deth_circle( omega_n_12_output, rmsh.c_r ) * (1.0 / rmsh.r) * rmsh.ds_circle )}", \
        }] )
    csvfile_F.flush()

