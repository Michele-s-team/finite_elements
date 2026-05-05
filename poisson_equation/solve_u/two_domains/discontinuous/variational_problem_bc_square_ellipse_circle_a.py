'''
solve for the Poisson equation on a domain given by a square with an ellipse and a circle in it, where the region beteen the circle and the ellipse is meshed 
Here u obeys a Poisson equation in the square, and a different, nonlinear equation in the shape
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
    
class f_0_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 2 * (-1 + x[0]**2 + 10 * x[1]**2)

    def value_shape(self):
        return (1,)
    
class f_1_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 6.0

    def value_shape(self):
        return (1,)

class u_exact_0_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 1 + x[0] ** 2 - 2 * x[1] ** 2

    def value_shape(self):
        return (1,)
    
class u_exact_1_expression(UserExpression):
    def eval(self, values, x):

        # test case 1
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

    def value_shape(self):
        return (1,)
    


msh.interpolate_dg(fsp.u_exact, u_exact_0_expression(), rmsh.sf, rmsh.lmsh.parameters['sub_mesh_0_id'])
msh.interpolate_dg(fsp.u_exact, u_exact_1_expression(), rmsh.sf, rmsh.lmsh.parameters['sub_mesh_1_id'])

msh.interpolate_dg(fsp.f, f_0_expression(), rmsh.sf, rmsh.lmsh.parameters['sub_mesh_0_id'])
msh.interpolate_dg(fsp.f, f_1_expression(), rmsh.sf, rmsh.lmsh.parameters['sub_mesh_1_id'])



sub_mesh_0_label, sub_mesh_1_label = msh.plus_minus(rmsh.lmsh.mesh, rmsh.sf, rmsh.lmsh.parameters["sub_mesh_0_id"], rmsh.lmsh.parameters["sub_mesh_1_id"], rmsh.dS_ellipse)

print(f'label_0 ={sub_mesh_0_label}\nlabel_1 = {sub_mesh_1_label}')


bcs = []

# I assign a value to the function to give a reasonable initial condition to the solver
fsp.u.assign(Constant(rpam.parameters['u_0']))
# fsp.u.assign(fsp.u_exact)



# variational functional for the original problem (poisson equation)
F_0 =   msh.ufl_conditional_form(rmsh.lmsh.mesh,
                                rmsh.sf,
                                fsp.u * fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f * fsp.nu_u,
                                fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f * fsp.nu_u,
                                rmsh.lmsh.parameters['sub_mesh_0_id'],
                                rmsh.lmsh.parameters['sub_mesh_1_id']
                                ) * rmsh.dx \
        - bgeo.facet_normal[i] * fsp.u * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_circle \
        - bgeo.facet_normal(sub_mesh_0_label)[i] * fsp.u(sub_mesh_0_label) * ((fsp.u(sub_mesh_0_label)).dx(i)) * (fsp.nu_u(sub_mesh_0_label)) * rmsh.dS_ellipse \
        - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_lrtb \
        - bgeo.facet_normal(sub_mesh_1_label)[i] * ((fsp.u(sub_mesh_1_label)).dx(i)) * (fsp.nu_u(sub_mesh_1_label)) * rmsh.dS_ellipse


F_I =   - msh.average(fsp.u.dx(i)) * msh.jump(fsp.u * fsp.nu_u, bgeo.facet_normal)[i] * rmsh.dS_I[0] \
        - msh.average(fsp.u.dx(i)) * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] * rmsh.dS_I[1] \
        + rpam.parameters['alpha']/rmsh.r_mesh * (\
            ( msh.jump(fsp.u, bgeo.facet_normal)[i] * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] ) * (rmsh.dS_I[0] + rmsh.dS_I[1]) \
            )


F_b =   rpam.parameters['alpha']/rmsh.r_mesh *(\
            (fsp.u - fsp.u_exact) * fsp.nu_u * rmsh.ds_lrtb + \
            (fsp.u(sub_mesh_1_label) - fsp.u_exact(sub_mesh_1_label)) * fsp.nu_u(sub_mesh_1_label) * rmsh.dS_ellipse + \
            (fsp.u - fsp.u_exact) * fsp.nu_u * rmsh.ds_circle + \
            (fsp.u(sub_mesh_0_label) - fsp.u_exact(sub_mesh_0_label)) * fsp.nu_u(sub_mesh_0_label) * rmsh.dS_ellipse \
        )


F = F_0 + F_I + F_b

