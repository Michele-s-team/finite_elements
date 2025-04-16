from fenics import *
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import input_output as io
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

# Create XDMF files for visualization output
xdmffile_v = XDMFFile((rarg.args.output_directory) + '/v.xdmf')
xdmffile_w = XDMFFile((rarg.args.output_directory) + '/w.xdmf')
xdmffile_sigma = XDMFFile((rarg.args.output_directory) + '/sigma.xdmf')
xdmffile_z = XDMFFile((rarg.args.output_directory) + '/z.xdmf')
xdmffile_omega = XDMFFile((rarg.args.output_directory) + '/omega.xdmf')
xdmffile_mu = XDMFFile((rarg.args.output_directory) + '/mu.xdmf')
xdmffile_nu = XDMFFile((rarg.args.output_directory) + '/nu.xdmf')
xdmffile_tau = XDMFFile((rarg.args.output_directory) + '/tau.xdmf')

xdmffile_f = XDMFFile((rarg.args.output_directory) + '/f.xdmf')
xdmffile_f.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_d = XDMFFile((rarg.args.output_directory) + '/d.xdmf')
xdmffile_d.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_dFdl = XDMFFile((rarg.args.output_directory) + '/dFdl.xdmf')
xdmffile_dFdl.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_dFdl_kappa_t = XDMFFile((rarg.args.output_directory) + '/dFdl_kappa_t.xdmf')
xdmffile_dFdl_kappa_t.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_dFdl_kappa_n = XDMFFile((rarg.args.output_directory) + '/dFdl_kappa_n.xdmf')
xdmffile_dFdl_kappa_n.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
v_output, w_output, sigma_output, z_output, omega_output, mu_output = fsp.psi.split(deepcopy=True)

# print solution to file
xdmffile_v.write(v_output, 0)
xdmffile_w.write(w_output, 0)
xdmffile_sigma.write(sigma_output, 0)
xdmffile_z.write(z_output, 0)
xdmffile_omega.write(omega_output, 0)
xdmffile_mu.write(mu_output, 0)

xdmffile_nu.write(fsp.nu, 0)
xdmffile_tau.write(fsp.tau, 0)

# print to csv file
io.print_vector_to_csvfile(v_output, (rarg.args.output_directory) + '/v.csv')
io.print_scalar_to_csvfile(w_output, (rarg.args.output_directory) + '/w.csv')
io.print_scalar_to_csvfile(sigma_output, (rarg.args.output_directory) + '/sigma.csv')
io.print_scalar_to_csvfile(z_output, (rarg.args.output_directory) + '/z.csv')
io.print_vector_to_csvfile(omega_output, (rarg.args.output_directory) + '/omega.csv')
io.print_scalar_to_csvfile(mu_output, (rarg.args.output_directory) + '/mu.csv')

io.print_vector_to_csvfile(fsp.nu, (rarg.args.output_directory) + '/nu.csv')
io.print_scalar_to_csvfile(fsp.tau, (rarg.args.output_directory) + '/tau.csv')

io.print_nodal_values_vector_to_csvfile(v_output, bgeo.mesh, (rarg.args.output_directory) + '/nodal_values/v.csv')
io.print_nodal_values_scalar_to_csvfile(w_output, bgeo.mesh, (rarg.args.output_directory) + '/nodal_values/w.csv')
io.print_nodal_values_scalar_to_csvfile(sigma_output, bgeo.mesh,
                                        (rarg.args.output_directory) + '/nodal_values/sigma.csv')
io.print_nodal_values_scalar_to_csvfile(z_output, bgeo.mesh, (rarg.args.output_directory) + '/nodal_values/z.csv')
io.print_nodal_values_vector_to_csvfile(omega_output, bgeo.mesh,
                                        (rarg.args.output_directory) + '/nodal_values/omega.csv')
io.print_nodal_values_scalar_to_csvfile(mu_output, bgeo.mesh, (rarg.args.output_directory) + '/nodal_values/mu.csv')

io.print_nodal_values_vector_to_csvfile(fsp.nu, bgeo.mesh, (rarg.args.output_directory) + '/nodal_values/nu.csv')
io.print_nodal_values_scalar_to_csvfile(fsp.tau, bgeo.mesh, (rarg.args.output_directory) + '/nodal_values/tau.csv')

# write the solutions in .h5 format so it can be read from other codes
HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/v.h5", "w").write(v_output, "/f")
HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/w.h5", "w").write(w_output, "/f")
HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/sigma.h5", "w").write(sigma_output, "/f")
HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/z.h5", "w").write(z_output, "/f")
HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/omega.h5", "w").write(omega_output, "/f")
HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/mu.h5", "w").write(mu_output, "/f")

HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/nu.h5", "w").write(fsp.nu, "/f")
HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/tau.h5", "w").write(fsp.tau, "/f")

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
# write tangential force due to viscosity and surface tension
xdmffile_dFdl.write(project(
    phys.dFdl_eta_sigma_t(v_output, w_output, omega_output, sigma_output, vp.eta, geo.n_c_r(bgeo.mesh, rmsh.c_r, omega_output)),
    fsp.Q_dFfl_t), 0)
io.print_vector_to_csvfile(project(
    phys.dFdl_eta_sigma_t(v_output, w_output, omega_output, sigma_output, vp.eta, geo.n_c_r(bgeo.mesh, rmsh.c_r, omega_output)),
    fsp.Q_dFfl_t), (rarg.args.output_directory) + '/dFdl.csv')

# write tangential force due to bending rigidity
xdmffile_dFdl_kappa_t.write(
    project(phys.dFdl_kappa_t(fsp.mu, vp.kappa, geo.n_c_r(bgeo.mesh, rmsh.c_r, omega_output)), fsp.Q_dFfl_t), 0)
io.print_vector_to_csvfile(
    project(phys.dFdl_kappa_t(fsp.mu, vp.kappa, geo.n_c_r(bgeo.mesh, rmsh.c_r, omega_output)), fsp.Q_dFfl_t),
    (rarg.args.output_directory) + '/dFdl_kappa_t.csv')

# write normal force due to bending rigidity
xdmffile_dFdl_kappa_n.write(
    project(phys.dFdl_kappa_n(fsp.mu, vp.kappa, geo.n_c_r(bgeo.mesh, rmsh.c_r, omega_output)), fsp.Q_dFfl_n), 0)
io.print_scalar_to_csvfile(
    project(phys.dFdl_kappa_n(fsp.mu, vp.kappa, geo.n_c_r(bgeo.mesh, rmsh.c_r, omega_output)), fsp.Q_dFfl_n),
    (rarg.args.output_directory) + '/dFdl_kappa_n.csv')


