'''
here 'a' anb 'b' refer to the two domains
    - 'a' is the label for the left square
    - 'b' is the label for the right square
'''


from fenics import *
import importlib
import numpy as np
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
        # values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2 + rpam.parameters['A'] * (x[0]-rmsh.lmsh.parameters['L_m'])

        # test case 2
        # values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 4 + rpam.parameters['A'] * (x[0]-rmsh.lmsh.parameters['L_m'])

        # test case 3
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 4 + rpam.parameters['A'] * np.sin(2 * np.pi / rmsh.lmsh.parameters['L'] * (x[0] - rmsh.lmsh.parameters['L_m'])) * np.cos(4 * np.pi * (x[0] + x[1]) / rmsh.lmsh.parameters['L'])

    def value_shape(self):
        return (1,)


class u_exact_r_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        # values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

        # test case 2
        # value`s[0] = 1 + x[0] ** 2 + 2 * x[1] ** 4 

        # test case 3
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 4 

    def value_shape(self):
        return (1,)



class d_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        # values[0] = rpam.parameters['A']

        # test case 2
        # values[0] = rpam.parameters['A']

        # test case 3
        values[0] = (2 * rpam.parameters['A'] * np.pi * np.cos((4 * np.pi * (rmsh.lmsh.parameters['L_m'] + x[1])) / rmsh.lmsh.parameters['L'])) / rmsh.lmsh.parameters['L']

    def value_shape(self):
        return (1,)


# build a UFL expression that contains the expression for the laplacians in both surface_l and surface_r

x_  = SpatialCoordinate(rmsh.lmsh.mesh)


'''
# test case 1
f = 6.0
'''

'''
# test case 2
f = 2.0 + 24.0 * x_[1]**2
'''



# test case 3
L = Constant(rmsh.lmsh.parameters['L'])
L_m = Constant(rmsh.lmsh.parameters['L_m'])
A   = Constant(rpam.parameters['A'])
pi = Constant(np.pi)

f = conditional(le(x_[0], L_m),
                    2.0 + 24.0 * x_[1]**2 + 2 * A * pi**2/ L**2 * (13 * sin((2 * pi * (L_m - 3 * x_[0] - 2 * x_[1])) / L) +  5 * sin((2 * pi * (L_m + x_[0] + 2 * x_[1])) / L) ),
                    2.0 + 24.0 * x_[1]**2)



fsp.u_exact_l.interpolate(u_exact_l_expression(element=fsp.Q.ufl_element()))
fsp.u_exact_r.interpolate(u_exact_r_expression(element=fsp.Q.ufl_element()))

fsp.d.interpolate(d_expression(element=fsp.Q.ufl_element()))

bcs = []


# variational functional for the original problem (poisson equation)
F_0 =   (fsp.u.dx(i) * fsp.nu_u.dx(i)) * rmsh.dx + \
        (f * fsp.nu_u) * rmsh.dx + \
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
