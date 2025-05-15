from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import input_output as io
import load_2d_mesh as lmsh
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi

i, j, k, l = ufl.indices(4)


xdmffile_z = XDMFFile( (rarg.args.output_directory) + "/z.xdmf" )
xdmffile_omega = XDMFFile( (rarg.args.output_directory) + "/omega.xdmf" )
xdmffile_mu = XDMFFile( (rarg.args.output_directory) + "/mu.xdmf" )
xdmffile_rho = XDMFFile( (rarg.args.output_directory) + "/rho.xdmf" )
xdmffile_tau = XDMFFile( (rarg.args.output_directory) + "/tau.xdmf" )

xdmffile_check = XDMFFile( (rarg.args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )


xdmffile_z.write( fsp.z_output, 0 )
xdmffile_omega.write( fsp.omega_output, 0 )
xdmffile_mu.write( fsp.mu_output, 0 )
xdmffile_rho.write( fsp.rho_output, 0 )
xdmffile_tau.write( fsp.tau_output, 0 )

io.print_scalar_to_csvfile( fsp.z_output, (rarg.args.output_directory) + '/z.csv' )
io.print_vector_to_csvfile( fsp.omega_output, (rarg.args.output_directory) + '/omega.csv' )
io.print_scalar_to_csvfile( fsp.mu_output, (rarg.args.output_directory) + '/mu.csv' )
io.print_vector_to_csvfile( fsp.rho_output, (rarg.args.output_directory) + '/rho.csv' )
io.print_vector_to_csvfile( fsp.tau_output, (rarg.args.output_directory) + '/tau.csv' )


xdmffile_check.write( project( fsp.mu_output - fsp.mu_exact, fsp.Q_z ), 0 )
xdmffile_check.write( project( sqrt((fsp.rho_output[i] - fsp.rho_exact[i]) * (fsp.rho_output[i] - fsp.rho_exact[i])), fsp.Q_z ), 0 )
xdmffile_check.write( project( fsp.tau_output - fsp.f, fsp.Q_z ), 0 )
xdmffile_check.close()

