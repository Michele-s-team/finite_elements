from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import parameters.read.solution as rpam
import solution_paths as solpath
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
psi_output, omega_output, rho_output, zeta_output = fsp.phi.split(deepcopy=True)

io.full_print(psi_output, 'psi', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, lmsh.mesh,
              'scalar')
io.full_print(omega_output, 'omega', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'scalar')
io.full_print(rho_output, 'rho', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'scalar')
io.full_print(zeta_output, 'zeta', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'scalar')

io.full_print(fsp.sigma, 'sigma', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'scalar')


io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + "metadata.csv", \
                                io.merge_dictionaries(rmsh.parameters, rpam.parameters))
