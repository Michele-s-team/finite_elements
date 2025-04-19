from fenics import *
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import input_output as io
import solution_paths as solpath
import physics as phys
import runtime_arguments as rarg

# CHANGE VARIATIONAL PROBLEM OR MESH HERE
import read_mesh_ring as rmsh
# import read_mesh_square as rmsh

# CHANGE VARIATIONAL PROBLEM OR MESH HERE
import variational_problem_bc_ring_1 as vp
# import variational_problem_bc_ring_2 as vp
# import variational_problem_bc_square_a as vp
# import variational_problem_bc_square_b as vp

i, j, k, l = ufl.indices(4)

xdmffile_f = XDMFFile((rarg.args.output_directory) + '/f.xdmf')
xdmffile_f.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_d = XDMFFile((rarg.args.output_directory) + '/d.xdmf')
xdmffile_d.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
v_output, w_output, sigma_output, z_output, omega_output, mu_output = fsp.psi.split(deepcopy=True)

io.full_print(v_output, 'v', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, bgeo.mesh,
              'vector')
io.full_print(w_output, 'w', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, bgeo.mesh,
              'scalar')
io.full_print(sigma_output, 'sigma', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
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

# print to file the forces which appear in the RHS of the equations
# tangential forces
xdmffile_f.write(project(phys.fvisc_t(fsp.d, omega_output, vp.eta), fsp.Q_f_t), 0)
xdmffile_f.write(project(phys.fsigma_t(sigma_output, omega_output), fsp.Q_f_t), 0)
xdmffile_f.write(
    project(phys.conv_cn_t(v_output, v_output, v_output, w_output, w_output, omega_output, vp.rho), fsp.Q_f_t), 0)

io.print_vector_to_csvfile(project(phys.fvisc_t(fsp.d, omega_output, vp.eta), fsp.Q_f_t),
                           (rarg.args.output_directory) + '/fvisc_t.csv')
io.print_vector_to_csvfile(project(phys.fsigma_t(sigma_output, omega_output), fsp.Q_f_t),
                           (rarg.args.output_directory) + '/fsigma_t.csv')
io.print_vector_to_csvfile(
    project(phys.conv_cn_t(v_output, v_output, v_output, w_output, w_output, omega_output, vp.rho), fsp.Q_f_t),
    (rarg.args.output_directory) + '/fv_t.csv')

# normal forces
xdmffile_f.write(project(phys.fvisc_n(v_output, w_output, omega_output, fsp.mu, vp.eta), fsp.Q_f_n), 0)
xdmffile_f.write(project(phys.fel_n(omega_output, mu_output, fsp.tau, vp.kappa), fsp.Q_f_n), 0)
xdmffile_f.write(project(phys.flaplace(sigma_output, omega_output), fsp.Q_f_n), 0)
xdmffile_f.write(
    project(phys.conv_cn_n(v_output, v_output, v_output, w_output, w_output, omega_output, vp.rho), fsp.Q_f_n), 0)

io.print_scalar_to_csvfile(project(phys.fvisc_n(v_output, w_output, omega_output, fsp.mu, vp.eta), fsp.Q_f_n),
                           (rarg.args.output_directory) + '/fvisc_n.csv')
io.print_scalar_to_csvfile(project(phys.fel_n(omega_output, mu_output, fsp.tau, vp.kappa), fsp.Q_f_n),
                           (rarg.args.output_directory) + '/fel_n.csv')
io.print_scalar_to_csvfile(project(phys.flaplace(sigma_output, omega_output), fsp.Q_f_n),
                           (rarg.args.output_directory) + '/flaplace.csv')
io.print_scalar_to_csvfile(
    project(phys.conv_cn_n(v_output, v_output, v_output, w_output, w_output, omega_output, vp.rho), fsp.Q_f_n),
    (rarg.args.output_directory) + '/conv_cn_n.csv')

# print rate of deformation tensor to file
xdmffile_d.write(project(fsp.d, fsp.Q_d), 0)

# print residual of the PDEs to files
xdmffile_check = XDMFFile((rarg.args.output_directory) + "/check.xdmf")
xdmffile_check.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_check.write(project((geo.Nabla_v(v_output, omega_output)[i, i] - 2.0 * mu_output * w_output), fsp.Q_sigma), 0)
xdmffile_check.write(project( \
    sqrt((phys.fvisc_t(fsp.d, omega_output, vp.eta)[i] + phys.fsigma_t(sigma_output, omega_output)[i] -
          phys.conv_cn_t(v_output, v_output, v_output, w_output, w_output, omega_output, vp.rho)[i]) \
         * (phys.fvisc_t(fsp.d, omega_output, vp.eta)[i] + phys.fsigma_t(sigma_output, omega_output)[i] -
            phys.conv_cn_t(v_output, v_output, v_output, w_output, w_output, omega_output, vp.rho)[i])), \
    fsp.Q_f_n), 0)
xdmffile_check.write(project(phys.fvisc_n(v_output, w_output, omega_output, mu_output, vp.eta) \
                             + phys.fel_n(omega_output, mu_output, fsp.tau, vp.kappa) \
                             + phys.flaplace(sigma_output, omega_output) \
                             - phys.conv_cn_n(v_output, v_output, v_output, w_output, w_output, omega_output, vp.rho) \
                             , fsp.Q_f_n), 0)
xdmffile_check.write(
    project(project(sqrt((omega_output[i] - (z_output.dx(i))) * (omega_output[i] - (z_output.dx(i)))), fsp.Q_z),
            fsp.Q_z), 0)
xdmffile_check.write(project(project(mu_output - geo.H(omega_output), fsp.Q_z), fsp.Q_z), 0)

xdmffile_check.write(
    project(project(sqrt((fsp.nu[i] - (mu_output.dx(i))) * (fsp.nu[i] - (mu_output.dx(i)))), fsp.Q_z), fsp.Q_z), 0)
xdmffile_check.write(
    project(project(fsp.tau - geo.g_c(omega_output)[i, j] * geo.Nabla_f(fsp.nu, omega_output)[i, j], fsp.Q_z),
            fsp.Q_tau), 0)
xdmffile_check.write(project(project((geo.d(v_output, w_output, omega_output)[i, j] - fsp.d[i, j]) * (
        geo.d(v_output, w_output, omega_output)[i, j] - fsp.d[i, j]), fsp.Q_z), fsp.Q_tau), 0)

# write to file forces per unit length

io.full_print(
    project(phys.dFdl_eta_sigma_t(v_output, w_output, omega_output, sigma_output, vp.eta,
                                  geo.n_c_r(bgeo.mesh, rmsh.c_r, omega_output)), fsp.Q_dFfl_t),
    'dFdl_eta_sigma_t', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,
    bgeo.mesh, 'vector')

io.full_print(
    project(phys.dFdl_kappa_t(fsp.mu, vp.kappa, geo.n_c_r(bgeo.mesh, rmsh.c_r, omega_output)), fsp.Q_dFfl_t),
    'dFdl_kappa_t', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,
    bgeo.mesh, 'vector')

io.full_print(
    project(phys.dFdl_kappa_n(fsp.mu, vp.kappa, geo.n_c_r(bgeo.mesh, rmsh.c_r, omega_output)), fsp.Q_dFfl_n),
    'dFdl_kappa_n', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,
    bgeo.mesh, 'scalar')

io.full_print(
    project(phys.dFdl_eta_sigma_3d(v_output, w_output, omega_output, sigma_output, vp.eta,
                                   geo.n_c_r(bgeo.mesh, rmsh.c_r, omega_output)), fsp.Q_3d),
    'dFdl_eta_sigma_3d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
    solpath.nodal_values_path, bgeo.mesh,
    'vector_3d')

io.full_print(
    project(
        phys.dFdl_kappa_3d(omega_output, mu_output, vp.kappa, geo.n_c_r(bgeo.mesh, rmsh.c_r, omega_output)), fsp.Q_3d),
    'dFdl_kappa_3d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,
    bgeo.mesh, 'vector_3d')

io.full_print(
    project( \
        phys.dFdl_tot_3d(v_output, w_output, omega_output, mu_output, sigma_output, vp.eta, vp.kappa,
                         geo.n_c_r(bgeo.mesh, rmsh.c_r, omega_output)), fsp.Q_3d),
    'dFdl_tot_3d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,
    bgeo.mesh, 'vector_3d')
