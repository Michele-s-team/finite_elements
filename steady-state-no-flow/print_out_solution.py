from fenics import *
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import input_output as io
import physics as phys
import solution_paths as solpath
import runtime_arguments as rarg

# CHANGE VARIATIONAL PROBLEM OR MESH HERE
# import read_mesh_square_no_circle as rmsh
# import read_mesh_square as rmsh

# CHANGE VARIATIONAL PROBLEM OR MESH HERE
import variational_problem_bc_ring as vp
# import variational_problem_bc_square_no_circle_a as vp
# import variational_problem_bc_square_a as vp
# import variational_problem_bc_square_b as vp

i, j, k, l = ufl.indices(4)

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
z_output, omega_output, mu_output = fsp.psi.split(deepcopy=True)

io.full_print(fsp.sigma, 'sigma', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              bgeo.mesh, 'scalar')
io.full_print(z_output, 'z', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, bgeo.mesh,
              'scalar')
io.full_print(omega_output, 'omega', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              bgeo.mesh, 'vector')
io.full_print(mu_output, 'mu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, bgeo.mesh,
              'scalar')

io.full_print(fsp.nu, 'nu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, bgeo.mesh,
              'vector')
io.full_print(fsp.tau, 'tau', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, bgeo.mesh,
              'scalar')

xdmffile_f = XDMFFile((rarg.args.output_directory) + '/f.xdmf')
xdmffile_f.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_f.write(project(phys.fel_n(omega_output, mu_output, fsp.tau, vp.kappa), fsp.Q_sigma), 0)
xdmffile_f.write(project(-phys.flaplace(fsp.sigma, omega_output), fsp.Q_sigma), 0)

xdmffile_check = XDMFFile((rarg.args.output_directory) + "/check.xdmf")
xdmffile_check.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_check.write(project(
    project(phys.fel_n(omega_output, mu_output, fsp.tau, vp.kappa) + phys.flaplace(fsp.sigma, omega_output), fsp.Q_z),
    fsp.Q_z), 0)
xdmffile_check.write(
    project(project(sqrt((omega_output[i] - (z_output.dx(i))) * (omega_output[i] - (z_output.dx(i)))), fsp.Q_z),
            fsp.Q_z), 0)
xdmffile_check.write(project(project(mu_output - geo.H(omega_output), fsp.Q_z), fsp.Q_z), 0)
xdmffile_check.write(
    project(project(sqrt((fsp.nu[i] - (mu_output.dx(i))) * (fsp.nu[i] - (mu_output.dx(i)))), fsp.Q_z), fsp.Q_z), 0)
xdmffile_check.write(
    project(project(fsp.tau - geo.g_c(omega_output)[i, j] * geo.Nabla_f(fsp.nu, omega_output)[i, j], fsp.Q_z),
            fsp.Q_tau), 0)
