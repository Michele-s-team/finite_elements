'''
    this module solves the variational problem

    U^{n-1/2} - U^{n-3/2} = dt * (\vec{v} \cdot \hat{n}(U^{n-1/2})) \hat{n}(U^{n-1/2})

    with periodic BCs u(x_l) = u(x_r)
'''


from fenics import *
import importlib
import numpy as np
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



class ys_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rmsh.lmsh.parameters['c'][0] + rmsh.lmsh.parameters['r'] * np.cos(2 * np.pi * x[0] / rmsh.lmsh.mesh_parameters[1]['L'])
        values[1] = rmsh.lmsh.parameters['c'][1] + rmsh.lmsh.parameters['r'] * np.sin(2 * np.pi * x[0] / rmsh.lmsh.mesh_parameters[1]['L'])
    
    def value_shape(self):
        return (2,)


class psi_0_expression(UserExpression):
    def eval(self, values, x):

        values[0] = -2*np.pi*x[0]/rmsh.lmsh.mesh_parameters[1]['L']
    
    def value_shape(self):
        return (1,)


fsp.ys.interpolate(ys_expression(element=fsp.Q_U.ufl_element()))
fsp.psi_0.interpolate(psi_0_expression(element=fsp.Q_psi_0.ufl_element()))


# no BCs are needed here: the periodic BC is already implemented through the periodicity of the function space
bcs_U = [ ]
bcs_nu_and_dpsi = [ ]
bcs_mu = [ ]

# variational functional for the original problem (first-order equation equation)
F_U = (fsp.U_n_12[alpha] - fsp.U_n_32[alpha] - dt * fsp.v_disk_n_1_0_0_on_1[alpha]) * \
    (
        fsp.nu_U[alpha]
    ) * rmsh.dx_mesh[1]
 

F_nu_psi = (
    ((fsp.ys[0] + fsp.U_n_12[0]).dx(0) - geo.e(fsp.psi_0 + fsp.dpsi_n_12, fsp.nu_n_12)[0, 0])
    * (-cos(fsp.psi_0 + fsp.dpsi_n_12) * fsp.nu_nu + fsp.nu_n_12 * sin(fsp.psi_0 + fsp.dpsi_n_12) * fsp.nu_dpsi)
    + ((fsp.ys[1] + fsp.U_n_12[1]).dx(0) - geo.e(fsp.psi_0 + fsp.dpsi_n_12, fsp.nu_n_12)[0, 1])
    * (sin(fsp.psi_0 + fsp.dpsi_n_12) * fsp.nu_nu + fsp.nu_n_12 * cos(fsp.psi_0 + fsp.dpsi_n_12) * fsp.nu_dpsi)
) * geo.sqrt_detg(fsp.psi_0 + fsp.dpsi_n_12, fsp.nu_n_12) * rmsh.dx_mesh[1]

F_mu = ((geo.H(fsp.psi_0 + fsp.dpsi_n_12, fsp.nu_n_12) - fsp.mu_n_12) * fsp.nu_mu) \
       * geo.sqrt_detg(fsp.psi_0 + fsp.dpsi_n_12, fsp.nu_n_12) * rmsh.dx_mesh[1]
