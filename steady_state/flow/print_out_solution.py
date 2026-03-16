from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.load as lmsh
import physics as phys
import parameters.read.solution as rpam
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


i, j, k, l = ufl.indices(4)

xdmffile_f = XDMFFile((rarg.args.output_directory) + '/f.xdmf')
xdmffile_f.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_d = XDMFFile((rarg.args.output_directory) + '/d.xdmf')
xdmffile_d.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
v_output, w_output, sigma_output, z_output, omega_output, mu_output = fsp.psi.split(deepcopy=True)

io.full_print(v_output, 'v', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(w_output, 'w', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(sigma_output, 'sigma', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(z_output, 'z', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(omega_output, 'omega', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(mu_output, 'mu', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)

io.full_print(fsp.tau, 'tau', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)

# print to file the forces which appear in the RHS of the equations
# tangential forces
xdmffile_f.write(project(phys.fvisc_t(fsp.d, omega_output, rpam.parameters['eta']), fsp.Q_f_t), 0)
xdmffile_f.write(project(phys.fsigma_t(sigma_output, omega_output), fsp.Q_f_t), 0)
xdmffile_f.write(
    project(phys.conv_cn_t(v_output, v_output, v_output, w_output, w_output, omega_output, rpam.parameters['rho']), fsp.Q_f_t), 0)

io.print_vector_to_csvfile(project(phys.fvisc_t(fsp.d, omega_output, rpam.parameters['eta']), fsp.Q_f_t),
                           (rarg.args.output_directory) + '/fvisc_t.csv')
io.print_vector_to_csvfile(project(phys.fsigma_t(sigma_output, omega_output), fsp.Q_f_t),
                           (rarg.args.output_directory) + '/fsigma_t.csv')
io.print_vector_to_csvfile(
    project(phys.conv_cn_t(v_output, v_output, v_output, w_output, w_output, omega_output, rpam.parameters['rho']), fsp.Q_f_t),
    (rarg.args.output_directory) + '/fv_t.csv')

# normal forces
xdmffile_f.write(project(phys.fvisc_n(v_output, w_output, omega_output, fsp.mu, rpam.parameters['eta']), fsp.Q_f_n), 0)
xdmffile_f.write(project(phys.fel_n(omega_output, mu_output, fsp.tau, rpam.parameters['kappa']), fsp.Q_f_n), 0)
xdmffile_f.write(project(phys.flaplace(sigma_output, omega_output), fsp.Q_f_n), 0)
xdmffile_f.write(
    project(phys.conv_cn_n(v_output, v_output, v_output, w_output, w_output, omega_output, rpam.parameters['rho']), fsp.Q_f_n), 0)

io.print_scalar_to_csvfile(project(phys.fvisc_n(v_output, w_output, omega_output, fsp.mu, rpam.parameters['eta']), fsp.Q_f_n),
                           (rarg.args.output_directory) + '/fvisc_n.csv')
io.print_scalar_to_csvfile(project(phys.fel_n(omega_output, mu_output, fsp.tau, rpam.parameters['kappa']), fsp.Q_f_n),
                           (rarg.args.output_directory) + '/fel_n.csv')
io.print_scalar_to_csvfile(project(phys.flaplace(sigma_output, omega_output), fsp.Q_f_n),
                           (rarg.args.output_directory) + '/flaplace.csv')
io.print_scalar_to_csvfile(
    project(phys.conv_cn_n(v_output, v_output, v_output, w_output, w_output, omega_output, rpam.parameters['rho']), fsp.Q_f_n),
    (rarg.args.output_directory) + '/conv_cn_n.csv')

# print rate of deformation tensor to file
xdmffile_d.write(project(fsp.d, fsp.Q_d), 0)

# print residual of the PDEs to files
xdmffile_check = XDMFFile((rarg.args.output_directory) + "/check.xdmf")
xdmffile_check.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_check.write(project((geo.Nabla_v(v_output, omega_output)[i, i] - 2.0 * mu_output * w_output), fsp.Q_sigma), 0)
xdmffile_check.write(project( \
    sqrt((phys.fvisc_t(fsp.d, omega_output, rpam.parameters['eta'])[i] + phys.fsigma_t(sigma_output, omega_output)[i] -
          phys.conv_cn_t(v_output, v_output, v_output, w_output, w_output, omega_output, rpam.parameters['rho'])[i]) \
         * (phys.fvisc_t(fsp.d, omega_output, rpam.parameters['eta'])[i] + phys.fsigma_t(sigma_output, omega_output)[i] -
            phys.conv_cn_t(v_output, v_output, v_output, w_output, w_output, omega_output, rpam.parameters['rho'])[i])), \
    fsp.Q_f_n), 0)
xdmffile_check.write(project(phys.fvisc_n(v_output, w_output, omega_output, mu_output, rpam.parameters['eta']) \
                             + phys.fel_n(omega_output, mu_output, fsp.tau, rpam.parameters['kappa']) \
                             + phys.flaplace(sigma_output, omega_output) \
                             - phys.conv_cn_n(v_output, v_output, v_output, w_output, w_output, omega_output, rpam.parameters['rho']) \
                             , fsp.Q_f_n), 0)
xdmffile_check.write(
    project(project(sqrt((omega_output[i] - (z_output.dx(i))) * (omega_output[i] - (z_output.dx(i)))), fsp.Q_z),
            fsp.Q_z), 0)
xdmffile_check.write(project(project(mu_output - geo.H(omega_output), fsp.Q_z), fsp.Q_z), 0)

xdmffile_check.write(
    project(
        - phys.conv_cn_n(v_output, v_output, v_output, w_output, w_output, omega_output, rpam.parameters['rho']) \
        + phys.lhs_force_balance_equation(rpam.parameters['kappa'], omega_output, mu_output, sigma_output, fsp.tau) \
        + phys.fvisc_n(v_output, w_output, omega_output, mu_output, rpam.parameters['eta']),
        fsp.Q_tau), 0)
xdmffile_check.write(project(project((geo.d(v_output, w_output, omega_output)[i, j] - fsp.d[i, j]) * (
        geo.d(v_output, w_output, omega_output)[i, j] - fsp.d[i, j]), fsp.Q_z), fsp.Q_tau), 0)

# write to file forces per unit length

io.full_print(
    project(phys.dFdl_eta_sigma_t(v_output, w_output, omega_output, sigma_output, rpam.parameters['eta'],
                                  geo.n_c_r(lmsh.mesh, rmsh.parameters["c_r"][:2], omega_output)), fsp.Q_dFfl_t),
    'dFdl_eta_sigma_t', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path)

io.full_print(
    project(phys.dFdl_kappa_t(fsp.mu, rpam.parameters['kappa'], geo.n_c_r(lmsh.mesh, rmsh.parameters["c_r"][:2], omega_output)), fsp.Q_dFfl_t),
    'dFdl_kappa_t', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path)

io.full_print(
    project(phys.dFdl_kappa_n(fsp.mu, rpam.parameters['kappa'], geo.n_c_r(lmsh.mesh, rmsh.parameters["c_r"][:2], omega_output)), fsp.Q_dFfl_n),
    'dFdl_kappa_n', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path)

io.full_print(
    project(phys.dFdl_eta_sigma_3d(v_output, w_output, omega_output, sigma_output, rpam.parameters['eta'],
                                   geo.n_c_r(lmsh.mesh, rmsh.parameters["c_r"][:2], omega_output)), fsp.Q_3d),
    'dFdl_eta_sigma_3d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
    solpath.nodal_values_path)

io.full_print(
    project(
        phys.dFdl_kappa_3d(omega_output, mu_output, rpam.parameters['kappa'], geo.n_c_r(lmsh.mesh, rmsh.parameters["c_r"][:2], omega_output)), fsp.Q_3d),
    'dFdl_kappa_3d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path)

io.full_print(
    project( \
        phys.dFdl_tot_3d(v_output, w_output, omega_output, mu_output, sigma_output, rpam.parameters['eta'], rpam.parameters['kappa'],
                         geo.n_c_r(lmsh.mesh, rmsh.parameters["c_r"][:2], omega_output)), fsp.Q_3d),
    'dFdl_tot_3d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path)


io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + "metadata.csv", \
                                io.merge_dictionaries(rmsh.parameters, rpam.parameters))
