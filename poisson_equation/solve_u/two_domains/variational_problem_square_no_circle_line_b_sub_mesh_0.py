from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function as fu
import runtime_arguments as rarg
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


# exact expression for mesh 0: here I choose an expression for u_exact which matches fsp.u[1] on the top edge of mesh[0]
class u_exact_mesh_0_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2
  
      
    def value_shape(self):
        return (1,)


class grad_u_exact_mesh_0_expression(UserExpression):
    def eval(self, values, x):
        
        values[0] = 2.0 * x[0]
        values[1] = 4.0 * x[1]
   
    def value_shape(self):
        return (2,)


class laplacian_u_exact_mesh_0_expression(UserExpression):
    def eval(self, values, x):
        
        values[0] = 6.0

    def value_shape(self):
        return (1,)



fsp.u_exact[0].interpolate(u_exact_mesh_0_expression(element=fsp.Q[0].ufl_element()))
fsp.grad_u[0].interpolate(grad_u_exact_mesh_0_expression(element=fsp.V[0].ufl_element()))
fsp.f[0].interpolate(laplacian_u_exact_mesh_0_expression(element=fsp.Q[0].ufl_element()))


bcs = []

F = (fsp.u[0].dx(i) * fsp.nu_u[0].dx(i) + fsp.f[0] * fsp.nu_u[0]) * rmsh.dx_mesh[0] \
    - bgeo.facet_normal[0][i] * fsp.grad_u[0][i] * fsp.nu_u[0] * rmsh.ds_mesh[0]['ds']
