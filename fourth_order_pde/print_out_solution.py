from fenics import *
import ufl_legacy as ufl

import function_spaces as fsp
import input_output as io
import load_2d_mesh as lmsh
import runtime_arguments as rarg
import solution_paths as solpath

i, j, k, l = ufl.indices(4)

fsp.z_output, fsp.omega_output, fsp.mu_output, fsp.rho_output, fsp.tau_output = fsp.psi.split(deepcopy=True)

# xdmffile_z = XDMFFile((rarg.args.output_directory) + "/z.xdmf")
# xdmffile_omega = XDMFFile((rarg.args.output_directory) + "/omega.xdmf")
# xdmffile_mu = XDMFFile((rarg.args.output_directory) + "/mu.xdmf")
# xdmffile_rho = XDMFFile((rarg.args.output_directory) + "/rho.xdmf")
# xdmffile_tau = XDMFFile((rarg.args.output_directory) + "/tau.xdmf")

xdmffile_check = XDMFFile((rarg.args.output_directory) + "/check.xdmf")
xdmffile_check.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

# xdmffile_z.write(fsp.z_output, 0)
# xdmffile_omega.write(fsp.omega_output, 0)
# xdmffile_mu.write(fsp.mu_output, 0)
# xdmffile_rho.write(fsp.rho_output, 0)
# xdmffile_tau.write(fsp.tau_output, 0)

# io.print_scalar_to_csvfile(fsp.z_output, (rarg.args.output_directory) + '/z.csv')
# io.print_vector_to_csvfile(fsp.omega_output, (rarg.args.output_directory) + '/omega.csv')
# io.print_scalar_to_csvfile(fsp.mu_output, (rarg.args.output_directory) + '/mu.csv')
# io.print_vector_to_csvfile(fsp.rho_output, (rarg.args.output_directory) + '/rho.csv')
# io.print_vector_to_csvfile(fsp.tau_output, (rarg.args.output_directory) + '/tau.csv')

io.full_print(fsp.z_output, 'z', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'scalar')
io.full_print(fsp.omega_output, 'omega', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'vector')
io.full_print(fsp.mu_output, 'mu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'scalar')
io.full_print(fsp.rho_output, 'rho', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'vector')
io.full_print(fsp.tau_output, 'tau', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'scalar')

xdmffile_check.write(project(fsp.mu_output - fsp.mu_exact, fsp.Q_z), 0)
xdmffile_check.write(project(sqrt((fsp.rho_output[i] - fsp.rho_exact[i]) * (fsp.rho_output[i] - fsp.rho_exact[i])), fsp.Q_z), 0)
xdmffile_check.write(project(fsp.tau_output - fsp.f, fsp.Q_z), 0)
xdmffile_check.close()
