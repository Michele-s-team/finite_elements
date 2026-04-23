'''
solve for the Poisson equation on a domain given by a square with a shape in it, where the shape is meshed inside
It allows for discontinuities of both u and grad u
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
class f_shape_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 6.0 + rpam.parameters['A'] * 4.0

    def value_shape(self):
        return (1,)
    
class f_square_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 6.0

    def value_shape(self):
        return (1,)

class u_exact_shape_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2 + rpam.parameters['A'] * ((x[0]-rmsh.lmsh.parameters['c'][0])**2 + (x[1]-rmsh.lmsh.parameters['c'][1])**2 - rmsh.lmsh.parameters['r']**2) + rpam.parameters['B']

    def value_shape(self):
        return (1,)
    
class u_exact_square_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2

    def value_shape(self):
        return (1,)
    
class d_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rpam.parameters['A'] * 2.0 * rmsh.lmsh.parameters['r']

    def value_shape(self):
        return (1,)
    
class e_expression(UserExpression):
    def eval(self, values, x):


        '''
        e is the jump in u: assuming that '+' = 'square' and '-' = 'shape', we have

            msh.jump(fsp.u, bgeo.facet_normal[0])[i] = 
            = n_square[i] u_square + n_shape[i] u _shape = 
            = n_+[i] ( 1 + x[0] ** 2 + 2 * x[1] ** 2) + n_-[i] (1 + x[0] ** 2 + 2 * x[1] ** 2 + rpam.parameters['A'] * ((x[0]-rmsh.lmsh.parameters['c'][0])**2 + (x[1]-rmsh.lmsh.parameters['c'][1])**2 - rmsh.lmsh.parameters['r']**2) + rpam.parameters['B']) = 
            = n_-[i] (rpam.parameters['A'] * ((x[0]-rmsh.lmsh.parameters['c'][0])**2 + (x[1]-rmsh.lmsh.parameters['c'][1])**2 - rmsh.lmsh.parameters['r']**2) + rpam.parameters['B']) = 

        thus I set 

        e = (rpam.parameters['A'] * ((x[0]-rmsh.lmsh.parameters['c'][0])**2 + (x[1]-rmsh.lmsh.parameters['c'][1])**2 - rmsh.lmsh.parameters['r']**2) + rpam.parameters['B'])

        and 

        msh.jump(fsp.u, bgeo.facet_normal[0])[i]  = fsp.e *  n_-[i]

        and this is the term that must appear in the variational problem
        '''

        values[0] = (rpam.parameters['A'] * ((x[0]-rmsh.lmsh.parameters['c'][0])**2 + (x[1]-rmsh.lmsh.parameters['c'][1])**2 - rmsh.lmsh.parameters['r']**2) + rpam.parameters['B'])

    def value_shape(self):
        return (1,)


msh.interpolate_dg(fsp.u_exact, u_exact_shape_expression(), rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_0_id'])
msh.interpolate_dg(fsp.u_exact, u_exact_square_expression(), rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_1_id'])

msh.interpolate_dg(fsp.f, f_shape_expression(), rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_0_id'])
msh.interpolate_dg(fsp.f, f_square_expression(), rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_1_id'])

msh.interpolate_dg(fsp.d, d_expression())
msh.interpolate_dg(fsp.e, e_expression())


'''
# test plus_minus - start
print(f'plus_minus = {msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]["dS_shape"])}')
# test plus_minus - end
'''
sub_mesh_0_0_label, sub_mesh_0_1_label = msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]["dS_shape"])

print(f'label_ shape ={sub_mesh_0_0_label}\nlabel_square = {sub_mesh_0_1_label}')

'''
n_shape = bgeo.field_facet_normal(bgeo.facet_normal[0]('-'), rmsh.lmsh.mesh[0], rmsh.ds_mesh[0]['dS_shape'], interior = True)
io.full_print(n_shape, 'n_shape', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              mesh_function=rmsh.lmsh.sf[0])
# 



'''

bcs = []


# variational functional for the original problem (poisson equation)
F_0 =   (fsp.u.dx(i) * fsp.nu_u.dx(i)) * rmsh.dx_mesh[0]['dx'] + \
        (fsp.f * fsp.nu_u) * rmsh.dx_mesh[0]['dx'] + \
        - bgeo.facet_normal[0][i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_mesh[0]['ds']

# here I put the average for d because d is the same on both sides (it is a jump)
F_a = - (msh.average(fsp.d)* msh.average(fsp.nu_u)) * rmsh.ds_mesh[0]['dS_shape']

F_I = (
        - msh.average(fsp.u.dx(i)) * msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i]
        ) * rmsh.ds_mesh[0]['dS_I'] + \
        rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
            ( msh.jump(fsp.u, bgeo.facet_normal[0])[i] * msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i] ) * (rmsh.ds_mesh[0]['dS_I_shape'] + rmsh.ds_mesh[0]['dS_I_square']) + \
            # ( msh.jump(fsp.u, bgeo.facet_normal[0])[i]  - msh.jump(fsp.u_exact, bgeo.facet_normal[0])[i] ) *  msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i] * rmsh.ds_mesh[0]['dS_shape']
            ( msh.jump(fsp.u, bgeo.facet_normal[0])[i]  - (msh.average(fsp.e) * ((bgeo.facet_normal[0])("-"))[i] ) ) *  msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i] * rmsh.ds_mesh[0]['dS_shape']
            )

F_b =   rpam.parameters['alpha']/rmsh.r_mesh[0] * (fsp.u - fsp.u_exact) * fsp.nu_u * rmsh.ds_mesh[0]['ds']


F = F_0 + F_I + F_a + F_b
