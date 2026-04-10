from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import input_output as io
import parameters.read.solution as rpam
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_shape_expression(UserExpression):
    def eval(self, values, x):

     values[0] = 1 + x[0]**2 + 2 * x[1]**2 + ((x[0] - rmsh.lmsh.parameters['c'][0])**2 + (x[1] - rmsh.lmsh.parameters['c'][1])**2)

    def value_shape(self):
        return (1,)
    

class u_exact_square_expression(UserExpression):
    def eval(self, values, x):

     values[0] = 1 + x[0]**2 + 2 * x[1]**2

    def value_shape(self):
        return (1,)
    

class grad_u_exact_shape_expression(UserExpression):
    def eval(self, values, x):

     values[0] = 2 * x[0] + 2 * (x[0] - rmsh.lmsh.parameters['c'][0])
     values[1] = 4 * x[1] + 2 * (x[1] - rmsh.lmsh.parameters['c'][1])

    def value_shape(self):
        return (1,)


class grad_u_exact_square_expression(UserExpression):
    def eval(self, values, x):

     values[0] = 2 * x[0] 
     values[1] = 4 * x[1]

    def value_shape(self):
        return (1,)


class laplacian_u_exact_shape_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 6 + 4

    def value_shape(self):
        return (1,)
    

class laplacian_u_exact_square_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 6

    def value_shape(self):
        return (1,)
    
class g_shape_expression(UserExpression):
    def eval(self, values, x):

     values[0] = 0

    def value_shape(self):
        return (1,)


fsp.u_exact_shape.interpolate(u_exact_shape_expression(element=fsp.Q.ufl_element()))
fsp.u_exact_square.interpolate(u_exact_square_expression(element=fsp.Q.ufl_element()))

fsp.grad_u_exact_shape.interpolate(grad_u_exact_shape_expression(element=fsp.V.ufl_element()))
fsp.grad_u_exact_square.interpolate(grad_u_exact_square_expression(element=fsp.V.ufl_element()))

fsp.f_shape.interpolate(laplacian_u_exact_shape_expression(element=fsp.Q.ufl_element()))
fsp.f_square.interpolate(laplacian_u_exact_square_expression(element=fsp.Q.ufl_element()))

fsp.g_shape.interpolate(g_shape_expression(element=fsp.Q.ufl_element()))

def smooth_facet_normal(mesh, dS, side):

    V = VectorFunctionSpace(mesh, "CG", 2)

    n = FacetNormal(mesh)

    u = TrialFunction(V)
    v = TestFunction(V)


    a = inner(u(side), v(side)) * dS
    l = inner(n(side), v(side)) * dS

    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()
    n_smooth = Function(V)

    solve(A, n_smooth.vector(), L)

    return n_smooth

n_smooth = smooth_facet_normal(rmsh.lmsh.mesh[0], rmsh.ds_mesh[0]['ds_shape'], '+')


io.full_print(n_smooth, 'n', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)

# it seems that '+' points into the shape, '-' outside the shape

# boundary conditions for sub_mesh[0][1]
bcs = [ \
    DirichletBC(fsp.Q, fsp.u_exact_square, rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]["line_l_id"]),\
    DirichletBC(fsp.Q, fsp.u_exact_square, rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]["line_r_id"]),\
    DirichletBC(fsp.Q, fsp.u_exact_square, rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]["line_t_id"]),\
    DirichletBC(fsp.Q, fsp.u_exact_square, rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]["line_b_id"])
    ]


F_0 = (fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f_shape * fsp.nu_u) * rmsh.dx_mesh[0]['dx_shape'] \
    + (fsp.u.dx(i) * fsp.nu_u.dx(i) + fsp.f_square * fsp.nu_u) * rmsh.dx_mesh[0]['dx_square']\
    - ((bgeo.facet_normal[0])('-'))[i] * (fsp.u('-').dx(i)) * fsp.nu_u('-') * rmsh.ds_mesh[0]['ds_shape']\
    - ((bgeo.facet_normal[0])('+'))[i] * (fsp.u('+').dx(i)) * fsp.nu_u('+') * rmsh.ds_mesh[0]['ds_shape']\
    - bgeo.facet_normal[0][i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_mesh[0]['ds']

F_N = rpam.parameters['alpha']/rmsh.r_mesh[0] * \
    ( 
        (((bgeo.facet_normal[0])('+'))[i] * (fsp.u('+').dx(i)) + ((bgeo.facet_normal[0])('-'))[i] * (fsp.u('-').dx(i))) \
        - (((bgeo.facet_normal[0])('+'))[i] * (fsp.u_exact_square('+').dx(i)) + ((bgeo.facet_normal[0])('-'))[i] * (fsp.u_exact_shape('-').dx(i)))\
    )\
    * ( ((bgeo.facet_normal[0])('+'))[j] * (fsp.nu_u('+').dx(j)) + ((bgeo.facet_normal[0])('-'))[j] * (fsp.nu_u('-').dx(j)))\
    * rmsh.ds_mesh[0]['ds_shape']

F = F_0 + F_N