'''
Here we solve for the PDE given in solve.py, by imposing both Dirichlet BCs on z at the left and right boundary, and Dirichlet conditions on omega and mu at the middle vertex.
'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

assigner = FunctionAssigner(fsp.Q, [fsp.Q_z, fsp.Q_omega, fsp.Q_mu])




class z_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = (x[0] ** 4) / 48.0
        
    def value_shape(self):
        return (1,)


class omega_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = (x[0] ** 3) / 12.0

    def value_shape(self):
        return (1,)


class mu_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = (7 * x[0] ** 6 ) / 576.0

    def value_shape(self):
        return (1,)


class rho_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = (7.0 * x[0] ** 5) / 96.0

    def value_shape(self):
        return (1,)


class f_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = (35.0 * x[0] ** 4 ) / 96.0

    def value_shape(self):
        return (1,)
    
    
    
class z_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        epsilon = 1e-1
        values[0] = (x[0] ** 4) / 48.0 + epsilon * np.cos(2.0 * np.pi * x[0] / (rmsh.parameters['x_r'] - rmsh.parameters['x_l']))
        
    def value_shape(self):
        return (1,)


class omega_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        epsilon = 1e-2
        values[0] =  (x[0] ** 3) / 12.0 + epsilon * np.cos(2.0 * np.pi * x[0] / (rmsh.parameters['x_r'] - rmsh.parameters['x_l']))
        

    def value_shape(self):
        return (1,)


class mu_0_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = 2 * (7 * x[0] ** 6 ) / 576.0

    def value_shape(self):
        return (1,)
    
    
fsp.z_exact.interpolate(z_exact_expression(element=fsp.Q_z.ufl_element()))
fsp.omega_exact.interpolate(omega_exact_expression(element=fsp.Q_omega.ufl_element()))
fsp.mu_exact.interpolate(mu_exact_expression(element=fsp.Q_mu.ufl_element()))

fsp.rho_exact.interpolate(rho_exact_expression(element=fsp.Q_rho.ufl_element()))
fsp.tau_exact.interpolate(f_exact_expression(element=fsp.Q_tau.ufl_element()))

fsp.f.interpolate(f_exact_expression(element=fsp.Q_z.ufl_element()))
    

# here is assign a wrong value to u (f) on purpose to see whether the solver conveges to the right solution
fsp.z_0.interpolate(z_0_expression(element=fsp.Q_z.ufl_element()))
fsp.omega_0.interpolate(omega_0_expression(element=fsp.Q_omega.ufl_element()))
fsp.mu_0.interpolate(mu_0_expression(element=fsp.Q_mu.ufl_element()))
assigner.assign(fsp.psi, [fsp.z_0, fsp.omega_0, fsp.mu_0])




# main variational problem

bc_z_l = DirichletBC(fsp.Q.sub(0), fsp.z_exact, rmsh.vf, rmsh.parameters['vertex_l_id'])
bc_z_r = DirichletBC(fsp.Q.sub(0), fsp.z_exact, rmsh.vf, rmsh.parameters['vertex_r_id'])
bc_omega_m = DirichletBC(fsp.Q.sub(1), fsp.omega_exact, rmsh.vf, rmsh.parameters['vertex_m_id'])
bc_mu_m = DirichletBC(fsp.Q.sub(2), fsp.mu_exact, rmsh.vf, rmsh.parameters['vertex_m_id'])

bcs = [bc_z_l, bc_z_r, bc_omega_m, bc_mu_m]



F_z = ((fsp.mu.dx(j)) * (fsp.nu_z.dx(j)) + fsp.f * fsp.nu_z) * rmsh.dx \
      - bgeo.facet_normal[j] * (fsp.mu.dx(j)) * fsp.nu_z * rmsh.ds

F_omega = (fsp.z * ((fsp.nu_omega[i]).dx(i)) + fsp.omega[i] * fsp.nu_omega[i]) * rmsh.dx \
          - bgeo.facet_normal[i] * fsp.z * fsp.nu_omega[i] * rmsh.ds

F_mu = (fsp.z * fsp.omega[i] * (fsp.nu_mu.dx(i)) + fsp.mu * fsp.nu_mu) * rmsh.dx \
       - bgeo.facet_normal[i] * fsp.z * fsp.omega[i] * fsp.nu_mu * rmsh.ds
       
F_N = rpam.parameters['alpha'] / rmsh.r_mesh * (bgeo.facet_normal[i] * fsp.omega[i] - bgeo.facet_normal[i] * fsp.omega_exact[i]) * bgeo.facet_normal[j] * fsp.nu_omega[j] * rmsh.ds

 
F = (F_omega + F_z + F_mu) + F_N


#post-processing variational problem
bc_pp_rho = DirichletBC(fsp.Q_pp.sub(0), fsp.rho_exact, rmsh.boundary)
bc_pp_tau = DirichletBC(fsp.Q_pp.sub(1), fsp.tau_exact, rmsh.boundary)

bcs_pp = [bc_pp_rho, bc_pp_tau]

F_rho = (fsp.mu * ((fsp.nu_rho[i]).dx(i)) + fsp.rho[i] * fsp.nu_rho[i]) * rmsh.dx \
        - bgeo.facet_normal[i] * fsp.mu * fsp.nu_rho[i] * rmsh.ds

F_tau = (fsp.tau * fsp.nu_tau + fsp.rho[i] * (fsp.nu_tau.dx(i))) * rmsh.dx \
        - bgeo.facet_normal[i] * fsp.rho[i] * fsp.nu_tau * rmsh.ds
        
F_pp = F_rho + F_tau



