from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.load as lmsh
import physics as phys
import parameters.read.solution as rpam
import solution_paths as solpath
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
z_output, omega_output, mu_output = fsp.psi.split(deepcopy=True)

io.full_print(fsp.sigma, 'sigma', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(z_output, 'z', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(omega_output, 'omega', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(mu_output, 'mu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)

io.full_print(fsp.tau, 'tau', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)

xdmffile_f = XDMFFile((rarg.args.output_directory) + '/f.xdmf')
xdmffile_f.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_f.write(project(phys.fel_n(omega_output, mu_output, fsp.tau, rpam.parameters["kappa"]), fsp.Q_sigma), 0)
xdmffile_f.write(project(-phys.flaplace(fsp.sigma, omega_output), fsp.Q_sigma), 0)

xdmffile_check = XDMFFile((rarg.args.output_directory) + "/check.xdmf")
xdmffile_check.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_check.write(project(
    project(phys.fel_n(omega_output, mu_output, fsp.tau, rpam.parameters["kappa"]) + phys.flaplace(fsp.sigma, omega_output), fsp.Q_z),
    fsp.Q_z), 0)
xdmffile_check.write(
    project(project(sqrt((omega_output[i] - (z_output.dx(i))) * (omega_output[i] - (z_output.dx(i)))), fsp.Q_z),
            fsp.Q_z), 0)
xdmffile_check.write(project(project(mu_output - geo.H(omega_output), fsp.Q_z), fsp.Q_z), 0)

xdmffile_check.write(
    project(project( phys.lhs_force_balance_equation(rpam.parameters["kappa"], omega_output, mu_output, fsp.sigma, fsp.tau)  , fsp.Q_z),
            fsp.Q_tau), 0)

io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + "metadata.csv", \
                                io.merge_dictionaries(rmsh.parameters, rpam.parameters))
