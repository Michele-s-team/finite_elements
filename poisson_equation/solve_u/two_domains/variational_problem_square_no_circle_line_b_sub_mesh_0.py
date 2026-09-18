'''
solve for a degenerate variational problem on mesh[0] where the solution u[0] is determined modulo a constant
the constant is fixed by pinning the solution u[0] on the bottom left vertex of the mesh
'''

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


'''
BC that pins u[0] to vertex tagged with rmsh.parameters["vertex_lb_id"]
this BC sets u[0](vertex_lb) = 0 and it removes the degeneracy in the variational problem 
'''
# coordinates of the bottom-left point of the mesh
x_lb = (rmsh.lmsh.mesh[0].coordinates()[rmsh.vf[0].array() ==  rmsh.parameters["vertex_lb_id"]])[0]
vertex_lb = CompiledSubDomain("near(x[0], x_lb_0) && near(x[1], x_lb_1)", x_lb_0=x_lb[0], x_lb_1=x_lb[1])
bc_vertex_lb = DirichletBC(fsp.Q[0], Constant(1.0), vertex_lb, method="pointwise")

bcs = [bc_vertex_lb]

F = (fsp.u[0].dx(i) * fsp.nu_u[0].dx(i) + fsp.f[0] * fsp.nu_u[0]) * rmsh.dx_mesh[0] \
    - bgeo.facet_normal[0][i] * fsp.grad_u[0][i] * fsp.nu_u[0] * rmsh.ds_mesh[0]['ds']
