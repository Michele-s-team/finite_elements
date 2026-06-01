from fenics import *
import ufl

import importlib
import input_output as io
import solution_paths as solpath
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l, m = ufl.indices(5)

mu_dummy, grad_u_dummy = fsp.psi.split( deepcopy=True )



io.full_print(fsp.n, 'n', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
io.full_print(fsp.e, 't', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
io.full_print(project(fsp.e[i]*fsp.e[i], fsp.Q_mu), 'g', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])

io.full_print(project(fsp.e[i].dx(j) * fsp.e[j] * fsp.n[i], fsp.Q_mu), 'b', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])



io.full_print(mu_dummy, 'mu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
io.full_print(grad_u_dummy, 'grad_u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
    