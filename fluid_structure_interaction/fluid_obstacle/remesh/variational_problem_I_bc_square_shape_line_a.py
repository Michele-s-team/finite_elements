'''
    this module solves the variational problem

    U^{n-1/2} - U^{n-3/2} = dt * (\vec{v} \cdot \hat{n}(U^{n-1/2})) \hat{n}(U^{n-1/2})

    with periodic BCs u(x_l) = u(x_r)
'''


from fenics import *
import importlib
import numpy as np
from scipy.interpolate import CubicSpline
import ufl as ufl

import command as cmd
import differential_geometry.manifold.geometry as geo
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

cmd.set_gauge('arc_length')


alpha, beta = ufl.indices(2)

dt = rpam.parameters['T'] / rpam.parameters['N']



# no BCs are needed here: the periodic BC is already implemented through the periodicity of the function space
bcs_U = [ ]
bcs_nu_and_dpsi = [ ]
bcs_mu = [ ]

# variational functional for the original problem (first-order equation equation)
F_U = (fsp.U_n_12[alpha] - fsp.U_n_32[alpha] - dt * fsp.v_disk_n_1_0_0_on_1[alpha]) * \
    (
        fsp.nu_U[alpha]
    ) * rmsh.dx_mesh[1]
 

#  build a smooth U_n_12 - start
def smooth_field_fourier(U, dof_coords, dofmap_x, dofmap_y, L, n_harmonics, target_field):
    U_vec = U.vector().get_local()
    s_x   = dof_coords[dofmap_x, 0]
    s_y   = dof_coords[dofmap_y, 0]
    U_x   = U_vec[dofmap_x]
    U_y   = U_vec[dofmap_y]

    def design_matrix(s):
        cols = [np.ones_like(s)]
        for k in range(1, n_harmonics + 1):
            cols.append(np.cos(2 * np.pi * k * s / L))
            cols.append(np.sin(2 * np.pi * k * s / L))
        return np.column_stack(cols)

    A_x = design_matrix(s_x)
    A_y = design_matrix(s_y)

    U_x_fit = A_x @ np.linalg.lstsq(A_x, U_x, rcond=None)[0]
    U_y_fit = A_y @ np.linalg.lstsq(A_y, U_y, rcond=None)[0]

    dof_values = target_field.vector().get_local()
    dof_values[dofmap_x] = U_x_fit
    dof_values[dofmap_y] = U_y_fit
    target_field.vector().set_local(dof_values)
    target_field.vector().apply("insert")


dof_coords = fsp.Q_U.tabulate_dof_coordinates()
dofmap_x   = fsp.Q_U.sub(0).dofmap().dofs()
dofmap_y   = fsp.Q_U.sub(1).dofmap().dofs()
L          = rmsh.lmsh.mesh_parameters[1]['L']
#  build a smooth U_n_12 - end


F_nu_psi = (
    ((fsp.ys[0] + fsp.U_n_12_smooth[0]).dx(0) - geo.e(fsp.psi_0 + fsp.dpsi_n_12, fsp.nu_n_12)[0, 0])
    * (-cos(fsp.psi_0 + fsp.dpsi_n_12) * fsp.nu_nu + fsp.nu_n_12 * sin(fsp.psi_0 + fsp.dpsi_n_12) * fsp.nu_dpsi)
    + ((fsp.ys[1] + fsp.U_n_12_smooth[1]).dx(0) - geo.e(fsp.psi_0 + fsp.dpsi_n_12, fsp.nu_n_12)[0, 1])
    * (sin(fsp.psi_0 + fsp.dpsi_n_12) * fsp.nu_nu + fsp.nu_n_12 * cos(fsp.psi_0 + fsp.dpsi_n_12) * fsp.nu_dpsi)
) * geo.sqrt_detg(fsp.psi_0 + fsp.dpsi_n_12, fsp.nu_n_12) * rmsh.dx_mesh[1]

F_mu = ((geo.H(fsp.psi_0 + fsp.dpsi_n_12, fsp.nu_n_12) - fsp.mu_n_12) * fsp.nu_mu) \
       * geo.sqrt_detg(fsp.psi_0 + fsp.dpsi_n_12, fsp.nu_n_12) * rmsh.dx_mesh[1]
