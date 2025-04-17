from fenics import *
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import input_output as io
import physics as phys
import solution_paths as solpath

import print_out_solution as prout

# write to file forces per unit length

io.full_print(
    project(phys.dFdl_sigma_t(fsp.sigma, geo.n_c_r(bgeo.mesh, prout.rmsh.c_r, prout.omega_output)), fsp.Q_dFfl_t),
    'dFdl_sigma_t', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,
    bgeo.mesh, 'vector')

io.full_print(
    project(phys.dFdl_kappa_t(fsp.mu, prout.vp.kappa, geo.n_c_r(bgeo.mesh, prout.rmsh.c_r, prout.omega_output)),
            fsp.Q_dFfl_t),
    'dFdl_kappa_t', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,
    bgeo.mesh, 'vector')

io.full_print(
    project(phys.dFdl_kappa_n(fsp.mu, prout.vp.kappa, geo.n_c_r(bgeo.mesh, prout.rmsh.c_r, prout.omega_output)),
            fsp.Q_dFfl_n),
    'dFdl_kappa_n', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,
    bgeo.mesh, 'scalar')

io.full_print(
    project(phys.dFdl_sigma_3d(prout.omega_output, fsp.sigma, geo.n_c_r(bgeo.mesh, prout.rmsh.c_r, prout.omega_output)),
            fsp.Q_3d),
    'dFdl_sigma_3d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
    solpath.nodal_values_path, bgeo.mesh,
    'vector_3d')

io.full_print(
    project(
        phys.dFdl_kappa_3d(prout.omega_output, prout.mu_output, prout.vp.kappa,
                           geo.n_c_r(bgeo.mesh, prout.rmsh.c_r, prout.omega_output)), fsp.Q_3d),
    'dFdl_kappa_3d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,
    bgeo.mesh, 'vector_3d')

io.full_print(
    project( \
        phys.dFdl_sigma_kappa_3d(prout.omega_output, prout.mu_output, fsp.sigma, prout.vp.kappa,
                                 geo.n_c_r(bgeo.mesh, prout.rmsh.c_r, prout.omega_output)), fsp.Q_3d),
    'dFdl_sigma_kappa_3d', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
    solpath.nodal_values_path,
    bgeo.mesh, 'vector_3d')
