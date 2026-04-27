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

        values[0] = 1

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



'''
# test plus_minus - start
print(f'plus_minus = {msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]["dS_shape"])}')
# test plus_minus - end
'''
sub_mesh_0_0_label, sub_mesh_0_1_label = msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]["dS_shape"])

print(f'label_ shape ={sub_mesh_0_0_label}\nlabel_square = {sub_mesh_0_1_label}')


bcs = []


# test ufl method - start
from fenics import conditional

# lift the cell tag into a DG0 function
V_dg0 = FunctionSpace(rmsh.lmsh.mesh[0], 'DG', 0)
cell_tag = Function(V_dg0)
cell_tag.vector()[:] = rmsh.sf[0].array()   # DG0 dofs are ordered by cell index

# build f as a pure UFL expression
f_ufl = conditional(
    ufl.eq(cell_tag, rmsh.lmsh.parameters['sub_mesh_0_0_id']),
    (fsp.u - fsp.u_exact) * fsp.nu_u,
    fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f * fsp.nu_u
) 
# test ufl method-end


# variational functional for the original problem (poisson equation)
F_0 =   f_ufl * rmsh.dx_mesh[0]['dx'] \
        - bgeo.facet_normal[0][i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_mesh[0]['ds'] \
        - bgeo.facet_normal[0](sub_mesh_0_1_label)[i] * ((fsp.u(sub_mesh_0_1_label)).dx(i)) * (fsp.nu_u(sub_mesh_0_1_label)) * rmsh.ds_mesh[0]['dS_shape']

F_I = (
        - msh.average(fsp.u.dx(i)) * msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i]
        ) * rmsh.ds_mesh[0]['dS_I_square'] + \
        rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
            ( msh.jump(fsp.u, bgeo.facet_normal[0])[i] * msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i] ) * rmsh.ds_mesh[0]['dS_I_square'] \
            )

F_b =   rpam.parameters['alpha']/rmsh.r_mesh[0] *(\
            (fsp.u - fsp.u_exact) * fsp.nu_u * rmsh.ds_mesh[0]['ds'] + \
            (fsp.u(sub_mesh_0_1_label) - fsp.u_exact(sub_mesh_0_1_label)) * fsp.nu_u(sub_mesh_0_1_label) * rmsh.ds_mesh[0]['dS_shape']\
        )


F = F_0 + F_I + F_b


# test print  u _exact - start
import input_output as io
import solution_paths as solpath

io.full_print(fsp.u_exact, 'u_exact', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              mesh_function=rmsh.lmsh.sf[0])
# test print u_exact - end