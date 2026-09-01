from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function as fu
import function_spaces as fsp
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


# exact expression for sub_mesh 0: here I choose an expression for u_exact which matches fsp.u[1] on the top edge of sub_mesh[0]
class u_exact_sub_mesh_0_expression(UserExpression):
    def eval(self, values, x):
        
        # test case 1
        values[0] = (1 + x[0]**2 + 2 * x[1]**2) * np.cos(2 * np.pi * x[1] / rmsh.parameters['curve_coordinates'][0][1])
    
    def value_shape(self):
        return (1,)


class grad_u_exact_sub_mesh_0_expression(UserExpression):
    def eval(self, values, x):
        
        # test case 1
        values[0] = 2 * x[0] * np.cos((2 * np.pi * x[1]) / rmsh.parameters['curve_coordinates'][0][1])
        values[1] = 4 * x[1] * np.cos((2 * np.pi * x[1]) / rmsh.parameters['curve_coordinates'][0][1]) - (2 * np.pi * (1 + x[0]**2 + 2 * x[1]**2) * np.sin((2 * np.pi * x[1]) / rmsh.parameters['curve_coordinates'][0][1])) / rmsh.parameters['curve_coordinates'][0][1]

    def value_shape(self):
        return (2,)


class laplacian_u_exact_sub_mesh_0_expression(UserExpression):
    def eval(self, values, x):
        
        # test case 1
        values[0] = -((2 * ((-3 * rmsh.parameters['curve_coordinates'][0][1]**2 + 2 * np.pi**2 * (1 + x[0]**2 + 2 * x[1]**2)) * np.cos((2 * np.pi * x[1]) / rmsh.parameters['curve_coordinates'][0][1]) + 8 * rmsh.parameters['curve_coordinates'][0][1] * np.pi * x[1] * np.sin((2 * np.pi * x[1]) / rmsh.parameters['curve_coordinates'][0][1]))) / rmsh.parameters['curve_coordinates'][0][1]**2)

    def value_shape(self):
        return (1,)


# v_expression is assigned from the values of fsp.u[1], to transfer the solution of problem on sub_mesh[1] to sub_mesh[0]
class v_Expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = ((fsp.u)[1])(x[0]) 

    def value_shape(self):
        return (1,)


fsp.u_exact[0].interpolate(u_exact_sub_mesh_0_expression(element=fsp.Q[0].ufl_element()))
fsp.grad_u[0].interpolate(grad_u_exact_sub_mesh_0_expression(element=fsp.V[0].ufl_element()))
fsp.f[0].interpolate(laplacian_u_exact_sub_mesh_0_expression(element=fsp.Q[0].ufl_element()))

# variational problem
# solve problem 0 by using the solution of problem 1 to specify the BCs
# set the BC at the interface between sub_mesh[0] and sub_mesh[1] according to the solution fsp.u[1] obtained above
# impose the BCs for problem on sub_mesh[0], on the t boundary of sub_mesh[0], in terms of fsp.u_1_on_0, and solve problem on sub_mesh[0]
# force reload vp to update bc[0], because u_1_on_0 has changed
fsp.v.interpolate(v_Expression(element=fsp.Q[1].ufl_element()))
# set u_1_on_0 to be equal to v = u[1]**2 + cos(2 pi (x[0] - h))**2 on the top edge of sub_mesh[1]
fu.transfer_sub_mesh_to_mesh(fsp.v, fsp.u_1_on_0, rarg.args.input_directory, rmsh.parameters['sub_mesh_1_id'])

bcs = [ \
    DirichletBC(fsp.Q[0], fsp.u_exact[0], rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_l_id"]), \
    DirichletBC(fsp.Q[0], fsp.u_exact[0], rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_r_id"]), \
    DirichletBC(fsp.Q[0], fsp.u_1_on_0, rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["sub_mesh_1_id"]), \
    DirichletBC(fsp.Q[0], fsp.u_exact[0], rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_b_id"])
]

F = (fsp.u[0].dx(i) * fsp.nu_u[0].dx(i) + fsp.f[0] * fsp.nu_u[0]) * rmsh.dx_sub_mesh[0] \
    - bgeo.sub_mesh_facet_normal[0][i] * (fsp.u[0].dx(i)) * fsp.nu_u[0] * rmsh.ds_sub_mesh[0]['ds']
