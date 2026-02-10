from fenics import *
import importlib
import numpy as np
import switch_problem as swi
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        values[0] = np.cos(2*np.pi*x[0]/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))

    def value_shape(self):
        return (1,)


class grad_u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = - 2*np.pi/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']) * np.sin(2*np.pi*x[0]/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))

    def value_shape(self):
        return (1,)


class f_expression(UserExpression):
    def eval(self, values, x):
        values[0] = - 2*np.pi/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']) * np.sin(2*np.pi*x[0]/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))

    def value_shape(self):
        return (1,)


class hess_u_exact_expression(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)

    def eval(self, values, x):
      
        # Matrix components
        values[0] = (2*np.pi/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))**2 * np.cos(2*np.pi*x[0]/(rmsh.parameters['x_r'] - rmsh.parameters['x_l']))


     
    def value_shape(self):
        return (1, 1)


fsp.u_exact.interpolate(u_exact_expression(element=fsp.Q.ufl_element()))
fsp.grad_u.interpolate(grad_u_expression(element=fsp.V.ufl_element()))
fsp.f.interpolate(f_expression(element=fsp.Q.ufl_element()))

fsp.hess_u_exact.interpolate(hess_u_exact_expression(element=fsp.T.ufl_element()))

# import a Dirichlet boundary condition on the left vertex, the BC on the right vertex is given by periodicity
bc_u_l = DirichletBC(fsp.Q, fsp.u_exact, rmsh.vf, rmsh.parameters['vertex_l_id'])
bcs = [bc_u_l]

# bcs=[]

# variational functional for the original problem (first-order equation equation)
F = (fsp.u.dx(0) - fsp.f) * fsp.nu_u.dx(0) * rmsh.dx
 
# variational functional for post-processing problem (pp) to obtain the hessian (hess)
F_pp = (fsp.hess_u[i, j] * fsp.nu_hess_u[i, j] + (fsp.u.dx(j)) * ((fsp.nu_hess_u[i, j]).dx(i))) * rmsh.dx \
    - (bgeo.facet_normal[i] * (fsp.u.dx(j)) * fsp.nu_hess_u[i, j]) * rmsh.ds
