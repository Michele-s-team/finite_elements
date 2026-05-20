from fenics import *
import importlib
import numpy as np

import mesh.utils as msh
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

# `c` are the coordinates of the center of mass of the elastic body

class y_expression(UserExpression):
    def eval(self, values, x):

        values[0] = x[0]
        values[1] = x[1]

    def value_shape(self):
        return (2,)

msh.interpolate_dg(fsp.y, y_expression())

# 1. compute c[i] = {c_i}_notes
c = [msh.average_wrt_measure(fsp.y[i], rmsh.dx_mesh[0]['dx_shape']) for i in range(2)]
print(f'*** c = {c}')

# 2. compute C[i][j]= {C_{ij}}_notes
C = [[msh.average_wrt_measure((fsp.y[i] + fsp.u_n[i]) * (fsp.y[j] - c[j]), rmsh.dx_mesh[0]['dx_shape']) for j in range(2)] for i in range(2)]

print(f'*** C = {C}')


# 3. compute t[i] = {t_i}_notes
t = [msh.average_wrt_measure(fsp.u_n[i], rmsh.dx_mesh[0]['dx_shape']) for i in range(2)]

print(f'*** t = {t}')

# 4. compute chi = \chi_notes and theta = \theta_notes
chi = (C[1][0] - C[0][1]) / (C[0][0] + C[1][1])
theta = np.arcsin(chi/sqrt(1.0 + chi**2))

print(f'*** chi = {chi}\ntheta = {theta}')