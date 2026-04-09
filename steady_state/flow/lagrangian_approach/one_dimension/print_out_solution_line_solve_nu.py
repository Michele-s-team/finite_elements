from fenics import *
import importlib
import ufl as ufl

import differential_geometry.manifold.geometry as geo
import input_output as io
import parameters.read.solution as rpam
import solution_paths as solpath
import runtime_arguments as rarg
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

v_output, w_output, sigma_output, psi_output, mu_output, u_output, nu_output = fsp.phi.split(deepcopy=True)

io.full_print(v_output, 'v', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(w_output, 'w', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(sigma_output, 'sigma', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(psi_output, 'psi', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(mu_output, 'mu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(u_output, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)

io.full_print(nu_output, 'nu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)

io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + "metadata.csv", \
                                io.merge_dictionaries(rmsh.parameters, rpam.parameters))
