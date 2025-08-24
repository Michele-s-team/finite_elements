from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import geometry as geo
import input_output as io
import load_mesh as lmsh
import physics as phys
import read_parameters_solve as rpam
import solution_paths as solpath
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

psi_output, mu_output, X_output = fsp.phi.split(deepcopy=True)

io.full_print(psi_output, 'psi', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, lmsh.mesh,
              'scalar')
io.full_print(mu_output, 'mu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'scalar')
io.full_print(X_output, 'X', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, lmsh.mesh,
              'vector')

io.full_print(fsp.sigma, 'sigma', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'scalar')

io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + "metadata.csv", \
                                io.merge_dictionaries(rmsh.parameters, rpam.parameters))
