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

    
class f_box_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 4.0

    def value_shape(self):
        return (1,)

class u_exact_surface_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 2

    def value_shape(self):
        return (1,)
    
class u_exact_box_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2 - x[2]**2

    def value_shape(self):
        return (1,)
    


msh.interpolate_dg(fsp.u_exact, u_exact_surface_expression(), rmsh.cf, rmsh.lmsh.parameters['surface_volume_id'])
msh.interpolate_dg(fsp.u_exact, u_exact_box_expression(), rmsh.cf, rmsh.lmsh.parameters['box_volume_id'])
msh.interpolate_dg(fsp.f, f_box_expression(), rmsh.cf, rmsh.lmsh.parameters['box_volume_id'])

bcs = []


# variational functional for the original problem (poisson equation)
F_0 =   msh.ufl_conditional_form(rmsh.lmsh.mesh,
                                rmsh.cf,
                                (fsp.u - fsp.u_exact) * fsp.nu_u,
                                fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f * fsp.nu_u,
                                rmsh.lmsh.parameters['surface_volume_id'],
                                rmsh.lmsh.parameters['box_volume_id']
                                ) * \
        rmsh.dx \
        - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds \
        - msh.side(bgeo.facet_normal, rmsh.I_box)[i] * ((msh.side(fsp.u, rmsh.I_box)).dx(i)) * (msh.side(fsp.nu_u, rmsh.I_box)) * rmsh.dS_surface

F_I = (
        - msh.average(fsp.u.dx(i)) * msh.jump(fsp.nu_u, bgeo.facet_normal)[i]
        ) * rmsh.dS_I_box + \
        rpam.parameters['alpha']/rmsh.r_mesh * (\
            ( msh.jump(fsp.u, bgeo.facet_normal)[i] * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] ) * rmsh.dS_I_box \
            )

F_b =   rpam.parameters['alpha']/rmsh.r_mesh *(\
            (fsp.u - fsp.u_exact) * fsp.nu_u * rmsh.ds + \
            (msh.side(fsp.u, rmsh.I_box) - msh.side(fsp.u_exact, rmsh.I_box)) * msh.side(fsp.nu_u, rmsh.I_box) * rmsh.dS_surface\
        )


F = F_0 + F_I + F_b
