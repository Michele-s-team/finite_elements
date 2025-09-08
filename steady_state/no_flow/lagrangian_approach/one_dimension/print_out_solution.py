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

psi_output, mu_output, X_output, nu_output = fsp.phi.split(deepcopy=True)

# print out the solution
io.full_print(psi_output, 'psi', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, lmsh.mesh,
              'scalar')
io.full_print(mu_output, 'mu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'scalar')
io.full_print(X_output, 'X', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, lmsh.mesh,
              'vector')

# print out the given fields of the surface tension and arc-length gauge
io.full_print(fsp.sigma, 'sigma', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'scalar')
io.full_print(nu_output, 'nu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'scalar')


# geo.e(fsp.psi, fsp.nu)[0, alpha]
io.full_print(project((fsp.X[0].dx(0) - geo.e(fsp.psi, fsp.nu)[0, 0]), fsp.Q_psi), 'err_1', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, lmsh.mesh,
              'scalar')

io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + "metadata.csv", \
                                io.merge_dictionaries(rmsh.parameters, rpam.parameters))
