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

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import mesh.utils as msh
import runtime_arguments as rarg
import parameters.read.solution as rpam
import switch_problem as swi


rmsh = importlib.import_module(swi.rmsh)

alpha, beta = ufl.indices(2)

dt = rpam.parameters['T'] / rpam.parameters['N']


class ys_expression(UserExpression):
    def eval(self, values, x):

        p = msh.map_1d_to_2d(x[0], rmsh.lmsh.mesh[0], rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]['shape_coordinates'], rmsh.lmsh.mesh_parameters[0]['shape_id'])

        values[0] = p[0]
        values[1] = p[1]

    def value_shape(self):
        return (2,)

fsp.ys.interpolate(ys_expression(element=fsp.Q_U.ufl_element()))


# no BCs are needed here: the periodic BC is already implemented through the periodicity of the function space
bcs=[ ]

# variational functional for the original problem (first-order equation equation)
F_U = (fsp.U_n_12[alpha] - fsp.U_n_32[alpha] - dt * (fsp.v_square_n_1_0_1_on_1[beta] * bgeo.n_ale(fsp.ys, fsp.U_n_12)[beta]) * bgeo.n_ale(fsp.ys, fsp.U_n_12)[alpha]) * \
    (
        fsp.nu_U[alpha] - \
        dt * (
            fsp.v_square_n_1_0_1_on_1[beta] * bgeo.delta_n_ale(fsp.ys, fsp.U_n_12, fsp.nu_U)[beta] * bgeo.n_ale(fsp.ys, fsp.U_n_12)[alpha] + \
            fsp.v_square_n_1_0_1_on_1[beta] * bgeo.n_ale(fsp.ys, fsp.U_n_12)[beta] * bgeo.delta_n_ale(fsp.ys, fsp.U_n_12, fsp.nu_U)[alpha]
        )
    ) * rmsh.dx_mesh[1]
 

