import csv
from fenics import *
import importlib
import os
import ufl

import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import input_output as io
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi


fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
sh = importlib.import_module(swi.sh)

i, j, k, l, m, n, o, p, q, r, s, t, u = ufl.indices(13)

mu_dummy, grad_u_dummy = fsp.psi.split( deepcopy=True )



# 1 print fields
io.full_print(project(as_tensor((fsp.f[i] + grad_u_dummy[i, k] * fsp.f[k]) , (i)), fsp.V), 'e_cur', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])

io.full_print(
    project(as_tensor(  
        (- sqrt( dot(fsp.f, fsp.f) / (ela.F(fsp.u)[p, q] * ela.F(fsp.u)[p, r] * fsp.f[q] * fsp.f[r] ) ) \
               * bgeo.epsilon[i, s] * ela.F(fsp.u)[s, t] * bgeo.epsilon[t, u] * fsp.nu[u] ), \
        (i)), fsp.V), \
    'n_cur', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])


io.full_print(project(as_tensor( (fsp.f[i] + fsp.grad_u[i, k] * fsp.f[k]).dx(j) * fsp.f[j], (i)), fsp.V), 'dot_e_cur', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])

io.full_print(\
    project( (fsp.f[i] + grad_u_dummy[i, k] * fsp.f[k]) * (fsp.f[i] + grad_u_dummy[i, l] * fsp.f[l]), fsp.Q_mu), \
    'g', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0]\
    )

io.full_print(project(fsp.f[i].dx(j) * fsp.f[j] * fsp.nu[i], fsp.Q_mu), 'b', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])


io.full_print(mu_dummy, 'mu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
io.full_print(grad_u_dummy, 'grad_u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
    



# 2 print the curve y_s
filename_y_s = rarg.args.output_directory + '/y_s_dy_ds.csv'
os.makedirs(os.path.dirname(filename_y_s), exist_ok=True)

csvfile = open(filename_y_s, 'a', newline='')
fieldnames = ['t', 'y_s:0', 'y_s:1', 'dy_ds:0', 'dy_ds:1']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

N = len(rmsh.parameters['shape_coordinates'])
M = N+1

for ii in range(M):

    ss = ii/(M-1)

    writer.writerows([{
        fieldnames[0]: \
            ss, \
        fieldnames[1]: \
            sh.y_s_dy_ds(ss)[0][0], \
        fieldnames[2]: \
            sh.y_s_dy_ds(ss)[0][1],\
        fieldnames[3]: \
            sh.y_s_dy_ds(ss)[1][0], \
        fieldnames[4]: \
            sh.y_s_dy_ds(ss)[1][1]
        }])

csvfile.close()
