from fenics import *
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import read_mesh_square as rmsh

i, j = ufl.indices( 2 )


class u_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2
        # values[0] = np.sin( 2 * (np.pi) * (x[0] + x[1]) ) * np.cos( 2 * (np.pi) * (x[0] - x[1]) ** 2 )

    def value_shape(self):
        return (1,)


class grad_u_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 2.0 * x[0]
        values[1] = 4.0 * x[1]
        # values[0] = 2 * (np.pi) * cos( 2 * (np.pi) * ((x[0]) - (x[1])) ** 2 ) * cos( 2 * (np.pi) * ((x[0]) + (x[1])) ) + 4 * (np.pi) * (-(x[0]) + (x[1])) * sin(
        #     2 * (np.pi) * ((x[0]) - (x[1])) ** 2 ) * sin( 2 * (np.pi) * ((x[0]) + (x[1])) )
        # values[1] = 2 * (np.pi) * cos( 2 * (np.pi) * ((x[0]) - (x[1])) ** 2 ) * cos( 2 * (np.pi) * ((x[0]) + (x[1])) ) + 4 * (np.pi) * ((x[0]) - (x[1])) * sin(
        #     2 * (np.pi) * ((x[0]) - (x[1])) ** 2 ) * sin( 2 * (np.pi) * ((x[0]) + (x[1])) )

    def value_shape(self):
        return (2,)


class laplacian_u_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 6.0
        # values[0] = 8 * (np.pi) * (-(np.pi) * (1 + 4 * (x[0] - (x[1])) ** 2) * cos( 2 * (np.pi) * (x[0] - (x[1])) ** 2 ) - sin( 2 * (np.pi) * (x[0] - (x[1])) ** 2 )) * sin(
        #     2 * (np.pi) * (x[0] + (x[1])) )

    def value_shape(self):
        return (1,)


class hess_u_exact_expression( UserExpression ):
    def init(self, **kwargs):
        super().init( **kwargs )

    def eval(self, values, x):
        values[0] = 2
        values[1] = 0
        values[2] = 0
        values[3] = 4
        # values[0] = 4 * np.pi * (
        #         4 * np.pi * (-x[0] + x[1]) * np.cos(2 * np.pi * (x[0] + x[1])) * np.sin(2 * np.pi * (x[0] - x[1])**2)
        #         - (np.pi * (1 + 4 * (x[0] - x[1])**2) * np.cos(2 * np.pi * (x[0] - x[1])**2)
        #         + np.sin(2 * np.pi * (x[0] - x[1])**2)) * np.sin(2 * np.pi * (x[0] + x[1]))
        #     )
        # values[1] =  4 * np.pi * (
        #         np.pi * (-1 + 4 * (x[0] - x[1])**2) * np.cos(2 * np.pi * (x[0] - x[1])**2)
        #         + np.sin(2 * np.pi * (x[0] - x[1])**2)
        #     ) * np.sin(2 * np.pi * (x[0] + x[1]))
        # values[2] = 4 * np.pi * (
        #         np.pi * (-1 + 4 * (x[0] - x[1])**2) * np.cos(2 * np.pi * (x[0] - x[1])**2)
        #         + np.sin(2 * np.pi * (x[0] - x[1])**2)
        #     ) * np.sin(2 * np.pi * (x[0] + x[1]))
        # values[3] = 4 * np.pi * (
        #         4 * np.pi * (x[0] - x[1]) * np.cos(2 * np.pi * (x[0] + x[1])) * np.sin(2 * np.pi * (x[0] - x[1])**2)
        #         - (np.pi * (1 + 4 * (x[0] - x[1])**2) * np.cos(2 * np.pi * (x[0] - x[1])**2)
        #         + np.sin(2 * np.pi * (x[0] - x[1])**2)) * np.sin(2 * np.pi * (x[0] + x[1]))
        #     )

    def value_shape(self):
        return (2, 2)


u_exact.interpolate( u_exact_expression( element=Q.ufl_element() ) )
grad_u.interpolate( grad_u_expression( element=V.ufl_element() ) )
f.interpolate( laplacian_u_expression( element=Q.ufl_element() ) )

hess_u_exact.interpolate( hess_u_exact_expression( element=T.ufl_element() ) )

bc_u = DirichletBC( Q, u_exact, boundary_tb )
bcs = [bc_u]

#variational functional for the original problem (poisson equation)
F = (dot( grad( u ), grad( nu_u ) ) + f * nu_u) * dx - dot( n, grad_u ) * nu_u * ds_lr - n[i] * (u.dx( i )) * nu_u * ds_tb

#variational functional for post-processing problem (pp) to obtain the hessian (hess)
F_pp = (hess_u[i, j] * nu_hess_u[i, j] + (u.dx( j )) * ((nu_hess_u[i, j]).dx( i ))) * dx \
       - (n[i] * (u.dx( j )) * nu_hess_u[i, j]) * ds
