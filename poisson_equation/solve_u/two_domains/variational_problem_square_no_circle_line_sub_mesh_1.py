from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


# exact expressions
class u_exact_sub_mesh_1_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 1 + x[0]**2 + 2 * (rmsh.parameters['curve_coordinates'][0][1])**2

    def value_shape(self):
        return (1,)


class grad_u_exact_sub_mesh_1_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 2 * x[0]

  
    def value_shape(self):
        return (1,)


class laplacian_u_exact_sub_mesh_1_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] =  2

    def value_shape(self):
        return (1,)


fsp.u_exact[1].interpolate(u_exact_sub_mesh_1_expression(element=fsp.Q[1].ufl_element()))
fsp.grad_u[1].interpolate(grad_u_exact_sub_mesh_1_expression(element=fsp.V[1].ufl_element()))
fsp.f[1].interpolate(laplacian_u_exact_sub_mesh_1_expression(element=fsp.Q[1].ufl_element()))

# sub_mesh[1]
# boundary conditions for sub_mesh[1]: constrain u[1] on the whole boundary of sub_mesh[1], i.e., on the ellipse and outer rectangle (lrtb)
bc_l = DirichletBC(fsp.Q[1], fsp.u_exact[1], rmsh.lmsh.mf_sub_meshes[1], rmsh.parameters['vertex_sub_mesh_1_l_id'])
bc_r = DirichletBC(fsp.Q[1], fsp.u_exact[1], rmsh.lmsh.mf_sub_meshes[1], rmsh.parameters['vertex_sub_mesh_1_r_id'])

bcs = [bc_l, bc_r]

F = (fsp.u[1].dx(i) * fsp.nu_u[1].dx(i) + fsp.f[1] * fsp.nu_u[1]) * rmsh.dx_sub_mesh[1] \
    - bgeo.sub_mesh_facet_normal[1][i] * (fsp.u[1].dx(i)) * fsp.nu_u[1] * rmsh.ds_sub_mesh[1]['ds']
