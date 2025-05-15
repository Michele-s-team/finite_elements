from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import input_output as io
import load_2d_mesh as lmsh
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi

xdmffile_z = XDMFFile( (args.output_directory) + "/z.xdmf" )
xdmffile_omega = XDMFFile( (args.output_directory) + "/omega.xdmf" )
xdmffile_mu = XDMFFile( (args.output_directory) + "/mu.xdmf" )
xdmffile_rho = XDMFFile( (args.output_directory) + "/rho.xdmf" )
xdmffile_tau = XDMFFile( (args.output_directory) + "/tau.xdmf" )

xdmffile_check = XDMFFile( (args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )


xdmffile_z.write( z_output, 0 )
xdmffile_omega.write( omega_output, 0 )
xdmffile_mu.write( mu_output, 0 )
xdmffile_rho.write( rho_output, 0 )
xdmffile_tau.write( tau_output, 0 )

io.print_scalar_to_csvfile( z_output, (args.output_directory) + '/z.csv' )
io.print_vector_to_csvfile( omega_output, (args.output_directory) + '/omega.csv' )
io.print_scalar_to_csvfile( mu_output, (args.output_directory) + '/mu.csv' )
io.print_vector_to_csvfile( rho_output, (args.output_directory) + '/rho.csv' )
io.print_vector_to_csvfile( tau_output, (args.output_directory) + '/tau.csv' )


xdmffile_check.write( project( mu_output - mu_exact, Q_z ), 0 )
xdmffile_check.write( project( sqrt((rho_output[i] - rho_exact[i]) * (rho_output[i] - rho_exact[i])), Q_z ), 0 )
xdmffile_check.write( project( tau_output - f, Q_z ), 0 )
xdmffile_check.close()

