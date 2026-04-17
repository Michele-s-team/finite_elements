'''
solve for a poisson equation by enforcing a jump in u and also in grad u

note: I don't understand why, but here I need to replace bgeo.facet_normal('+')[i] fsp.e with msh.jump(fsp.u_exact, bgeo.facet_normal)[i] for this code to give the correct solution. They are the same thing, so this is correct
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

'''
# print facet_normal('+') to check whether L = '+' or '-'
bgeo.field_facet_normal(bgeo.facet_normal('+'), rmsh.lmsh.mesh, rmsh.dS_m, interior=True)
'''


# test case 1
# 
import input_output as io
import solution_paths as solpath
import sys

class u_exact_l_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2 + rpam.parameters['A'] * np.cos(4*np.pi*(x[0]+x[1]-rmsh.lmsh.parameters['L_m'])/rmsh.lmsh.parameters['L']) + rpam.parameters['B']

    def value_shape(self):
        return (1,)
    
class u_exact_r_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

    def value_shape(self):
        return (1,)

class f_l_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 6.0

    def value_shape(self):
        return (1,)
    
class f_r_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 6.0

    def value_shape(self):
        return (1,)

msh.interpolate_dg(fsp.u_exact, u_exact_l_expression(), rmsh.sf, rmsh.lmsh.parameters['l_surface_id'])
msh.interpolate_dg(fsp.u_exact, u_exact_r_expression(), rmsh.sf, rmsh.lmsh.parameters['r_surface_id'])

msh.interpolate_dg(fsp.f, f_l_expression(), rmsh.sf, rmsh.lmsh.parameters['l_surface_id'])
msh.interpolate_dg(fsp.f, f_r_expression(), rmsh.sf, rmsh.lmsh.parameters['r_surface_id'])


io.full_print(fsp.u_exact, 'u_exact_l', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              mesh_function=rmsh.lmsh.sf)

sys.exit(1)
# 



def d_expression(x):
    return rpam.parameters['A']

def e_expression(x):
    return rpam.parameters['B']





msh.interpolate_dg(fsp.d, d_expression, rmsh.sf)
msh.interpolate_dg(fsp.e, e_expression, rmsh.sf)


bcs = []


# variational functional for the original problem (poisson equation)
F_0 =   (fsp.u.dx(i) * fsp.nu_u.dx(i)) * rmsh.dx + \
        (fsp.f * fsp.nu_u) * rmsh.dx - \
        bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds

# here I put the average for d because d is the same on both sides (it is a jump)
F_a = - (msh.average(fsp.d)* msh.average(fsp.nu_u)) * rmsh.dS_m


F_I = (
        - msh.average(fsp.u.dx(i)) * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] 
        ) * rmsh.dS +\
        rpam.parameters['alpha']/rmsh.r_mesh * ( \
            ( msh.jump(fsp.u, bgeo.facet_normal)[i] * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] ) * (rmsh.dS_l + rmsh.dS_r) + \
            ( msh.jump(fsp.u, bgeo.facet_normal)[i] - msh.jump(fsp.u_exact, bgeo.facet_normal)[i] ) * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] * rmsh.dS_m
        )

F_b =   rpam.parameters['alpha']/rmsh.r_mesh * (fsp.u - fsp.u_exact) * fsp.nu_u * rmsh.ds 


F = F_0 + F_I + F_a + F_b
