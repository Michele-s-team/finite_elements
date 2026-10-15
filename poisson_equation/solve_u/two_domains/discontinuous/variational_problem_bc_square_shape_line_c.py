'''
solve for the Poisson equation on a domain given by a square with a shape in it, where the shape is meshed inside
Here u obeys a Poisson equation in the square, and a non-differential equation u = d in the shape
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


# test case 1
    
class f_square_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 6.0

    def value_shape(self):
        return (1,)

class u_exact_shape_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 2

    def value_shape(self):
        return (1,)
    
class u_exact_square_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

    def value_shape(self):
        return (1,)
    


msh.interpolate_dg(fsp.u_exact, u_exact_shape_expression(), rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_0_id'])
msh.interpolate_dg(fsp.u_exact, u_exact_square_expression(), rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_1_id'])

msh.interpolate_dg(fsp.f, f_square_expression(), rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_1_id'])


bcs = []


# variational functional for the original problem (poisson equation)
F_0 =   msh.ufl_conditional_form(rmsh.lmsh.mesh[0],
                                rmsh.sf[0],
                                (fsp.u - fsp.u_exact) * fsp.nu_u,
                                fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f * fsp.nu_u,
                                rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                ) * \
        rmsh.dx_mesh[0]['dx'] \
        - bgeo.facet_normal[0][i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_mesh[0]['ds'] \
        - msh.side(bgeo.facet_normal[0], rmsh.I_sub_mesh_0_1)[i] * ((msh.side(fsp.u, rmsh.I_sub_mesh_0_1)).dx(i)) * (msh.side(fsp.nu_u, rmsh.I_sub_mesh_0_1)) * rmsh.ds_mesh[0]['dS_shape']

F_I = (
        - msh.average(fsp.u.dx(i)) * msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i]
        ) * rmsh.ds_mesh[0]['dS_I_square'] + \
        rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
            ( msh.jump(fsp.u, bgeo.facet_normal[0])[i] * msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i] ) * rmsh.ds_mesh[0]['dS_I_square'] \
            )

F_b =   rpam.parameters['alpha']/rmsh.r_mesh[0] *(\
            (fsp.u - fsp.u_exact) * fsp.nu_u * rmsh.ds_mesh[0]['ds'] + \
            (msh.side(fsp.u, rmsh.I_sub_mesh_0_1) - msh.side(fsp.u_exact, rmsh.I_sub_mesh_0_1)) * msh.side(fsp.nu_u, rmsh.I_sub_mesh_0_1) * rmsh.ds_mesh[0]['dS_shape']\
        )


F = F_0 + F_I + F_b
