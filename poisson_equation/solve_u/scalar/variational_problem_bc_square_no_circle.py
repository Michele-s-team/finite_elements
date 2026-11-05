from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

    def value_shape(self):
        return (1,)

    
class s_exact_expression(UserExpression):
    def eval(self, values, x):
        
        values[0] = 2.0

    def value_shape(self):
        return (1,)


class f_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 6.0/2.0

    def value_shape(self):
        return (1,)



fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q_u.ufl_element()))
fsp.s_exact.interpolate(s_exact_expression(element=fsp.Q_s.ufl_element()))
fsp.f.interpolate(f_expression(element=fsp.Q_u.ufl_element()))


bc_u = DirichletBC(fsp.Q.sub(0), fsp.u_exact, rmsh.boundary)

bcs = [bc_u]

F_u = (dot(grad(fsp.u), grad(fsp.nu_u)) + fsp.s * fsp.f * fsp.nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds \

F_s = (fsp.s/assemble(Constant(1.0) * rmsh.dx) - fsp.u) * fsp.nu_s * rmsh.dx

F = F_u + F_s