from fenics import *
import importlib
import numpy as np
import switch_problem as swi
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)

'''
class u_exact_mesh_1_expression(UserExpression):
    def eval(self, values, x):
        values[0] = np.sin(2*np.pi*x[0]**2/(rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l'])**2)

    def value_shape(self):
        return (1,)


class grad_u_mesh_1_expression(UserExpression):
    def eval(self, values, x):
        values[0] = (4*np.pi*x[0]*np.cos(2*np.pi*x[0]**2/(rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l'])**2))/(rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l'])**2

    def value_shape(self):
        return (1,)


class laplacian_u_mesh_1_expression(UserExpression):
    def eval(self, values, x):
        values[0] = (4*np.pi*((rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l'])**2*np.cos(2*np.pi*x[0]**2/(rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l'])**2) - 4*np.pi*x[0]**2*np.sin(2*np.pi*x[0]**2/(rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l'])**2)))/(rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l'])**4

    def value_shape(self):
        return (1,)


class hess_u_exact_mesh_1_expression(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)

    def eval(self, values, x):
      
        # Matrix components
        values[0] = (4*np.pi*((rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l'])**2*np.cos(2*np.pi*x[0]**2/(rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l'])**2) - 4*np.pi*x[0]**2*np.sin(2*np.pi*x[0]**2/(rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l'])**2)))/(rmsh.lmsh.mesh_parameters[1]['x_r'] - rmsh.lmsh.mesh_parameters[1]['x_l'])**4
     
    def value_shape(self):
        return (1, 1)


fsp.u_exact[1].interpolate(u_exact_mesh_1_expression(element=fsp.Q[1].ufl_element()))
fsp.grad_u[1].interpolate(grad_u_mesh_1_expression(element=fsp.V[1].ufl_element()))
fsp.f[1].interpolate(laplacian_u_mesh_1_expression(element=fsp.Q[1].ufl_element()))

fsp.hess_u_exact[1].interpolate(
    hess_u_exact_mesh_1_expression(element=fsp.T[1].ufl_element()))
'''

# impose a Dirichlet boundary condition on the left vertex in order to set C[1] -> 0
bc_l = DirichletBC(fsp.Q[1], Constant(-2.0 * (rmsh.lmsh.parameters['r'])**2), rmsh.mf[1], rmsh.lmsh.mesh_parameters[1]['vertex_l_id'])
bcs = [bc_l]

# variational functional
F = (fsp.u[1].dx(0) - fsp.u_0_1_on_1) *  fsp.nu_u[1].dx(0) * rmsh.dx_mesh[1]
