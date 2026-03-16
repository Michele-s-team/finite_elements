from fenics import *
import ufl as ufl

import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import runtime_arguments as rarg
import solution_paths as solpath

i, j, k, l = ufl.indices(4)

fsp.z_output, fsp.u_output, fsp.omega_output, fsp.mu_output = fsp.psi.split(deepcopy=True)

io.full_print(fsp.z_output, 'z', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'scalar')
io.full_print(fsp.u_output, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'scalar')
io.full_print(fsp.omega_output, 'omega', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'vector')
io.full_print(fsp.mu_output, 'mu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'scalar')
