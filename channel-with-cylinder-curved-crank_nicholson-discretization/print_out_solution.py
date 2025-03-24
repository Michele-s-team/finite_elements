import files as fi
import function_spaces as fsp
import input_output as io
import runtime_arguments as rarg
import boundary_geometry as bgeo


def print_z_omega():

    fi.xdmffile_z.write( fsp.z, 0 )
    fi.xdmffile_omega.write( fsp.omega, 0 )

    io.print_scalar_to_csvfile( fsp.z, (rarg.args.output_directory) + '/z.csv' )
    io.print_vector_to_csvfile( fsp.omega, (rarg.args.output_directory) + '/omega.csv' )

def print_solution(t, step, dt):

    fi.xdmffile_v.write( fsp.v_n, t )
    fi.xdmffile_v_.write( fsp.v_, t )
    fi.xdmffile_sigma.write( fsp.sigma_n_12, t - dt / 2.0 )
    fi.xdmffile_phi.write( fsp.phi, t )

    io.print_vector_to_csvfile( fsp.v_, (rarg.args.output_directory) + '/snapshots/csv/v_bar_' + str( step  ) + '.csv' )
    io.print_vector_to_csvfile( fsp.v_n, (rarg.args.output_directory) + '/snapshots/csv/v_n' + str( step  ) + '.csv' )
    io.print_scalar_to_csvfile( fsp.sigma_n_12, (rarg.args.output_directory) + '/snapshots/csv/sigma_n_12_' + str( step ) + '.csv' )
    io.print_scalar_to_csvfile( fsp.phi, (rarg.args.output_directory) + '/snapshots/csv/phi_' + str( step ) + '.csv' )

    io.print_nodal_values_vector_to_csvfile( fsp.v_, bgeo.mesh, (rarg.args.output_directory) + '/snapshots/csv/nodal_values/v_bar_' + str( step ) + '.csv' )
    io.print_nodal_values_vector_to_csvfile( fsp.v_n, bgeo.mesh, (rarg.args.output_directory) + '/snapshots/csv/nodal_values/v_n_' + str( step ) + '.csv' )
    io.print_nodal_values_scalar_to_csvfile( fsp.sigma_n_12, bgeo.mesh, (rarg.args.output_directory) + '/snapshots/csv/nodal_values/sigma_n_12_' + str( step ) + '.csv' )
    io.print_nodal_values_scalar_to_csvfile( fsp.phi, bgeo.mesh, (rarg.args.output_directory) + '/snapshots/csv/nodal_values/phi_' + str( step ) + '.csv' )