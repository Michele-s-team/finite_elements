'''
here 'a' anb 'b' refer to the two domains
    - 'a' is the label for the left square
    - 'b' is the label for the right square
'''


from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)

class u_exact_l_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        # values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2 + rpam.parameters['jump_coefficient'] * (x[0]-rmsh.lmsh.parameters['L_m'])

        # test case 2
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 4 + rpam.parameters['jump_coefficient'] * (x[0]-rmsh.lmsh.parameters['L_m'])

    def value_shape(self):
        return (1,)


class u_exact_r_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        # values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

        # test case 2
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 4

    def value_shape(self):
        return (1,)


class laplacian_u_l_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        # values[0] = 6.0

        # test case 2
        values[0] = 2.0 + 24.0 * x[1]**2

    def value_shape(self):
        return (1,)


class laplacian_u_r_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        # values[0] = 6.0

        # test case 2
        values[0] = 2.0 + 24.0 * x[1]**2

    def value_shape(self):
        return (1,)
    
class d_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = rpam.parameters['jump_coefficient']

    def value_shape(self):
        return (1,)


fsp.u_exact_l.interpolate(u_exact_l_expression(element=fsp.Q.ufl_element()))
fsp.u_exact_r.interpolate(u_exact_r_expression(element=fsp.Q.ufl_element()))

fsp.f_a.interpolate(laplacian_u_l_expression(element=fsp.Q.ufl_element()))
fsp.f_b.interpolate(laplacian_u_r_expression(element=fsp.Q.ufl_element()))

fsp.d.interpolate(d_expression(element=fsp.Q.ufl_element()))


bcs = []


# variational functional for the original problem (poisson equation)
F_0 =   (fsp.u.dx(i) * fsp.nu_u.dx(i)) * rmsh.dx + \
        (fsp.f_a * fsp.nu_u) * rmsh.dx_l + \
        (fsp.f_b * fsp.nu_u) * rmsh.dx_r +\
        - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds

# here I put the average for d because d is the same on both sides (it is a jump)
F_a = - (msh.average(fsp.d)* msh.average(fsp.nu_u)) * rmsh.dS_m

F_I = (
        - msh.average(fsp.u.dx(i)) * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] + \
        rpam.parameters['alpha']/rmsh.r_mesh * ( msh.jump(fsp.u, bgeo.facet_normal)[i] * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] )
        ) * rmsh.dS

F_b =   rpam.parameters['alpha']/rmsh.r_mesh * (\
            (fsp.u - fsp.u_exact_l) * fsp.nu_u * (rmsh.ds_l + rmsh.ds_lt + rmsh.ds_lb) + \
            (fsp.u - fsp.u_exact_r) * fsp.nu_u * (rmsh.ds_r + rmsh.ds_rt + rmsh.ds_rb)
        )


F = F_0 + F_I + F_a + F_b
