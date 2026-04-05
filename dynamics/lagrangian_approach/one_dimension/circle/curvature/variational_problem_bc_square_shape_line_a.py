'''
    this module solves the variational problem

    U^{n-1/2} - U^{n-3/2} = dt * (\vec{v} \cdot \hat{n}(U^{n-1/2})) \hat{n}(U^{n-1/2})

    with periodic BCs u(x_l) = u(x_r)
'''


from fenics import *
import importlib
import numpy as np
import os
import ufl as ufl

import command as cmd
import differential_geometry.manifold.geometry as geo
import function_spaces as fsp
import input_output as io
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)


cmd.set_gauge('arc_length')

mesh_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, '../', 'mesh_parameters.csv')) 


alpha, beta = ufl.indices(2)



class ys_expression(UserExpression):
    def eval(self, values, x):


        values[0] = mesh_parameters['c'][0] + mesh_parameters['r'] * np.cos(2 * np.pi * x[0] / rmsh.lmsh.mesh_parameters[1]['L'])
        values[1] = mesh_parameters['c'][1] + mesh_parameters['r'] * np.sin(2 * np.pi * x[0] / rmsh.lmsh.mesh_parameters[1]['L'])
    
    def value_shape(self):
        return (2,)
    
class psi0_expression(UserExpression):
    def eval(self, values, x):

        values[0] = -2*np.pi*x[0]/rmsh.lmsh.mesh_parameters[1]['L']
    
    def value_shape(self):
        return (1,)

fsp.ys.interpolate(ys_expression(element=fsp.Q_U.ufl_element()))
fsp.psi0.interpolate(psi0_expression(element=fsp.Q_psi0.ufl_element()))


bcs_nu_and_dpsi = [ ]
bcs_mu = [ ]


F_nu_psi = (
    ((fsp.ys[0] + fsp.U[0]).dx(0) - geo.e(fsp.psi0 + fsp.dpsi, fsp.nu)[0, 0])
    * (-cos(fsp.psi0 + fsp.dpsi) * fsp.nu_nu + fsp.nu * sin(fsp.psi0 + fsp.dpsi) * fsp.nu_dpsi)
    + ((fsp.ys[1] + fsp.U[1]).dx(0) - geo.e(fsp.psi0 + fsp.dpsi, fsp.nu)[0, 1])
    * (sin(fsp.psi0 + fsp.dpsi) * fsp.nu_nu + fsp.nu * cos(fsp.psi0 + fsp.dpsi) * fsp.nu_dpsi)
) * geo.sqrt_detg(fsp.psi0 + fsp.dpsi, fsp.nu) * rmsh.dx_mesh[1]

F_mu = ((geo.H(fsp.psi0 + fsp.dpsi, fsp.nu) - fsp.mu) * fsp.nu_mu) \
       * geo.sqrt_detg(fsp.psi0 + fsp.dpsi, fsp.nu) * rmsh.dx_mesh[1]