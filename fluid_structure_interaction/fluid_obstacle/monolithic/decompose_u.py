'''
decompose the displacement field u_n into its rigid part

NOTE: here we assume that -pi/2 < theta < pi/2
'''

from fenics import *
import importlib
import numpy as np

import calculus as cal
import function as fu
import mesh.utils as msh
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

'''
quantitties definitions

c[i] = {c_i}_{Decomposition of deformation field}
t[i] = {t_i}_{Decomposition of deformation field}
C[i][j] = {C_{ij}}_{Decomposition of deformation field}

chi = \chi_{Decomposition of deformation field}
'''

msh.interpolate_dg(fsp.y, fu.identity_expression())

# 1. compute c
c = [msh.average_wrt_measure(fsp.y[i], rmsh.dx_mesh[0]['dx_shape']) for i in range(2)]

# 2. compute C
C = [[msh.average_wrt_measure((fsp.y[i] + fsp.u_n[i]) * (fsp.y[j] - c[j]), rmsh.dx_mesh[0]['dx_shape']) for j in range(2)] for i in range(2)]

# 3. compute t
t = [msh.average_wrt_measure(fsp.u_n[i], rmsh.dx_mesh[0]['dx_shape']) for i in range(2)]

# 4. compute chi = \chi_notes and theta = \theta_notes
chi = (C[1][0] - C[0][1]) / (C[0][0] + C[1][1])
theta = np.arcsin(chi/np.sqrt(1.0 + chi**2))


# 5. compute phi_0

class phi_0_expression(UserExpression):
    def eval(self, values, x):

        result = cal.rotation_translation(x, theta, c, t)

        values[0] = result[0]
        values[1] = result[1]

    def value_shape(self):
        return (2,)

msh.interpolate_dg(fsp.phi_0, phi_0_expression(), rmsh.sf[0], rmsh.parameters['sub_mesh_0_0_id'])

