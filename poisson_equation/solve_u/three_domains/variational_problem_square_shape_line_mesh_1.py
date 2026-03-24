from fenics import *
import importlib
import numpy as np
import switch_problem as swi
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_mesh_1_expression(UserExpression):
    def eval(self, values, x):

        # test case 1) 
        # values[0] = (rmsh.lmsh.parameters['r']**2.0) * (-2.0 * np.cos(x[0] / rmsh.lmsh.parameters['r']) + np.sin(x[0] / rmsh.lmsh.parameters['r']))

        # test case 2) 
        values[0] = (1.0/12.0) * rmsh.lmsh.parameters['r']**4.0 * (-9.0 * np.cos(x[0] / rmsh.lmsh.parameters['r']) + np.cos((3.0 * x[0]) / rmsh.lmsh.parameters['r']) + 4.0 * (5.0 + np.cos((2.0 * x[0]) / rmsh.lmsh.parameters['r'])) * np.sin(x[0] / rmsh.lmsh.parameters['r']))

    def value_shape(self):
        return (1,)

fsp.u_exact[1].interpolate(u_exact_mesh_1_expression(element=fsp.Q[1].ufl_element()))


# impose a Dirichlet boundary condition on the left vertex in order to set C[1] -> 0
# test case 1
# bc_l = DirichletBC(fsp.Q[1], Constant(-2.0 * (rmsh.lmsh.parameters['r'])**2), rmsh.mf[1], rmsh.lmsh.mesh_parameters[1]['vertex_l_id'])
# test case 2
bc_l = DirichletBC(fsp.Q[1], Constant(-2.0/3.0 * (rmsh.lmsh.parameters['r'])**4), rmsh.mf[1], rmsh.lmsh.mesh_parameters[1]['vertex_l_id'])

bcs = [bc_l]

# variational functional
F = (fsp.u[1].dx(0) - fsp.u_0_1_on_1) *  fsp.nu_u[1].dx(0) * rmsh.dx_mesh[1]
