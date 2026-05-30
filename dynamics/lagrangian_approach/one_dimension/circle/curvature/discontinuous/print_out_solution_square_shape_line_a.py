from fenics import *
import ufl

import importlib
import input_output as io
import solution_paths as solpath
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l, m = ufl.indices(5)


io.full_print(fsp.n, 'n', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
io.full_print(fsp.t, 't', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
io.full_print(project(fsp.t[i]*fsp.t[i], fsp.Q), 'g', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])

io.full_print(project(fsp.t[i].dx(j) * fsp.t[j] * fsp.n[i], fsp.Q), 'b', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])



io.full_print(fsp.mu, 'mu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
    