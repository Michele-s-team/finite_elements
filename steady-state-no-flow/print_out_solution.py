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

# Create XDMF files for visualization output
xdmffile_z = XDMFFile((rarg.args.output_directory) + '/z.xdmf')
xdmffile_omega = XDMFFile((rarg.args.output_directory) + '/omega.xdmf')
xdmffile_mu = XDMFFile((rarg.args.output_directory) + '/mu.xdmf')

xdmffile_nu = XDMFFile((rarg.args.output_directory) + '/nu.xdmf')
xdmffile_tau = XDMFFile((rarg.args.output_directory) + '/tau.xdmf')

xdmffile_sigma = XDMFFile((rarg.args.output_directory) + '/sigma.xdmf')

xdmffile_f = XDMFFile((rarg.args.output_directory) + '/f.xdmf')
xdmffile_f.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

# print solution to file
xdmffile_z.write(z_output, 0)
xdmffile_omega.write(omega_output, 0)
xdmffile_mu.write(mu_output, 0)

xdmffile_nu.write(fsp.nu, 0)
xdmffile_tau.write(fsp.tau, 0)

xdmffile_sigma.write(fsp.sigma, 0)

# print to csv file
io.print_scalar_to_csvfile(z_output, (rarg.args.output_directory) + '/z.csv')
io.print_vector_to_csvfile(omega_output, (rarg.args.output_directory) + '/omega.csv')
io.print_scalar_to_csvfile(mu_output, (rarg.args.output_directory) + '/mu.csv')

io.print_vector_to_csvfile(fsp.nu, (rarg.args.output_directory) + '/nu.csv')
io.print_scalar_to_csvfile(fsp.tau, (rarg.args.output_directory) + '/tau.csv')

io.print_scalar_to_csvfile(fsp.sigma, (rarg.args.output_directory) + '/sigma.csv')

io.print_nodal_values_scalar_to_csvfile(z_output, bgeo.mesh, (rarg.args.output_directory) + '/nodal_values/z.csv')
io.print_nodal_values_vector_to_csvfile(omega_output, bgeo.mesh,
                                        (rarg.args.output_directory) + '/nodal_values/omega.csv')
io.print_nodal_values_scalar_to_csvfile(mu_output, bgeo.mesh, (rarg.args.output_directory) + '/nodal_values/mu.csv')

io.print_nodal_values_vector_to_csvfile(fsp.nu, bgeo.mesh, (rarg.args.output_directory) + '/nodal_values/nu.csv')
io.print_nodal_values_scalar_to_csvfile(fsp.tau, bgeo.mesh, (rarg.args.output_directory) + '/nodal_values/tau.csv')

# write the solutions in .h5 format so it can be read from other codes
HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/z.h5", "w").write(z_output, "/f")
HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/omega.h5", "w").write(omega_output, "/f")
HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/mu.h5", "w").write(mu_output, "/f")

HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/nu.h5", "w").write(fsp.nu, "/f")
HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/tau.h5", "w").write(fsp.tau, "/f")

HDF5File(MPI.comm_world, (rarg.args.output_directory) + "/h5/sigma.h5", "w").write(fsp.sigma, "/f")

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
