from fenics import *
import importlib
import ufl

import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import input_output as io
import solution_paths as solpath
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l, m, n, o, p, q, r, s, t, u = ufl.indices(13)

mu_dummy, grad_u_dummy = fsp.psi.split( deepcopy=True )




io.full_print(project(as_tensor((fsp.e[i] + grad_u_dummy[i, k] * fsp.e[k]) , (i)), fsp.V), 'e_cur', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])

io.full_print(
    project(as_tensor(  
        (- sqrt( dot(fsp.e, fsp.e) / (ela.F(fsp.u)[p, q] * ela.F(fsp.u)[p, r] * fsp.e[q] * fsp.e[r] ) ) \
               * bgeo.epsilon[i, s] * ela.F(fsp.u)[s, t] * bgeo.epsilon[t, u] * fsp.n[u] ), \
        (i)), fsp.V), \
    'n_cur', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])




io.full_print(project(as_tensor( (fsp.e[i] + fsp.grad_u[i, k] * fsp.e[k]).dx(j) * fsp.e[j], (i)), fsp.V), 'dot_e_cur', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])

io.full_print(\
    project( (fsp.e[i] + grad_u_dummy[i, k] * fsp.e[k]) * (fsp.e[i] + grad_u_dummy[i, l] * fsp.e[l]), fsp.Q_mu), \
    'g', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0]\
    )

io.full_print(project(fsp.e[i].dx(j) * fsp.e[j] * fsp.n[i], fsp.Q_mu), 'b', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])



io.full_print(mu_dummy, 'mu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
io.full_print(grad_u_dummy, 'grad_u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
    