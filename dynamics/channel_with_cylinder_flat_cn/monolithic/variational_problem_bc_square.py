from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import parameters.read.solution as rpam
import physics.fluid_mechanics as flu
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)


dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size

class v_l_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rpam.parameters['v_n_l_const'] * 4.0 * 1.5 * x[1] * (rmsh.parameters['h'] - x[1]) / (rmsh.parameters['h']**2)
        values[1] = 0.0

    def value_shape(self):
        return (2,)
    

class v_tb_circle_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)
    

class sigma_r_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rpam.parameters['sigma_r']

    def value_shape(self):
        return (1,)


fsp.v_l.interpolate(v_l_expression(element=fsp.Q_v_n.ufl_element()))
fsp.v_tb_circle.interpolate(v_tb_circle_expression(element=fsp.Q_v_n.ufl_element()))

fsp.sigma_r.interpolate(sigma_r_expression(element=fsp.Q_sigma_n.ufl_element()))


# boundary conditions for the surface_tension p
bcs = [
        DirichletBC(fsp.Q.sub(0), fsp.v_l, rmsh.mf, 2), 
        DirichletBC(fsp.Q.sub(0), fsp.v_tb_circle, rmsh.mf, 4), 
        DirichletBC(fsp.Q.sub(0), fsp.v_tb_circle, rmsh.mf, 5), 
        DirichletBC(fsp.Q.sub(0), fsp.v_tb_circle, rmsh.mf, 6),
        DirichletBC(fsp.Q.sub(1), fsp.sigma_r, rmsh.mf, 3)
    ]


# Define variational problem for step 1

F_v_n = ( \
                 rpam.parameters['rho'] * ((fsp.v_n[i] - fsp.v_n_1[i]) / dt + fsp.v_n[j] * (fsp.v_n[i]).dx(j)) * fsp.nu_v_n[i] \
                 + flu.sigma(fsp.v_n, fsp.sigma_n, rpam.parameters['mu'])[i, j] * (fsp.nu_v_n[i]).dx(j) \
         ) * rmsh.dx

# step 2
F_sigma_n = (fsp.v_n[i].dx(i)) * fsp.nu_sigma_n * rmsh.dx

F = F_v_n + F_sigma_n