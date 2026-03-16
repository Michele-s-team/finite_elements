from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function as fu
import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import parameters.read.solution as rpam
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)


class sigma_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters['sigma_const']

    def value_shape(self):
        return (1,)


class psi_exact_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class omega_exact_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class rho_exact_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class zeta_exact_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class psi_0_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = fsp.psi_0_read(x[0], x[1])

    def value_shape(self):
        return (1,)


class omega_0_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = fsp.omega_0_read(x[0], x[1])

    def value_shape(self):
        return (1,)


class rho_0_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = fsp.rho_0_read(x[0], x[1])

    def value_shape(self):
        return (1,)


class zeta_0_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = fsp.zeta_0_read(x[0], x[1])

    def value_shape(self):
        return (1,)


fsp.sigma.interpolate(sigma_Expression(element=fsp.Q_sigma.ufl_element()))

fsp.psi_0.interpolate(psi_exact_Expression(element=fsp.Q_psi.ufl_element()))
fsp.omega_0.interpolate(omega_exact_Expression(element=fsp.Q_omega.ufl_element()))
fsp.rho_0.interpolate(rho_exact_Expression(element=fsp.Q_rho.ufl_element()))
fsp.zeta_0.interpolate(zeta_exact_Expression(element=fsp.Q_zeta.ufl_element()))

fsp.psi_exact.interpolate(psi_exact_Expression(element=fsp.Q_psi.ufl_element()))
fsp.omega_exact.interpolate(omega_exact_Expression(element=fsp.Q_omega.ufl_element()))
fsp.rho_exact.interpolate(rho_exact_Expression(element=fsp.Q_rho.ufl_element()))
fsp.zeta_exact.interpolate(zeta_exact_Expression(element=fsp.Q_zeta.ufl_element()))

# uncomment this to set the initial profiles from the ODE soltion
######
print("Reading the initial profiles from file ...")
fu.set_from_file(fsp.psi_0_read, 'solution_ode/psi_ode.csv')
fsp.psi_0.interpolate(psi_0_Expression(element=fsp.Q_psi.ufl_element()))

fu.set_from_file(fsp.omega_0_read, 'solution_ode/omega_ode.csv')
fsp.omega_0.interpolate(omega_0_Expression(element=fsp.Q_omega.ufl_element()))

fu.set_from_file(fsp.rho_0_read, 'solution_ode/rho_ode.csv')
fsp.rho_0.interpolate(rho_0_Expression(element=fsp.Q_rho.ufl_element()))

fu.set_from_file(fsp.zeta_0_read, 'solution_ode/zeta_ode.csv')
fsp.zeta_0.interpolate(zeta_0_Expression(element=fsp.Q_zeta.ufl_element()))

# print out the read fields to file
io.full_print(fsp.psi_0, 'psi_0', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, lmsh.mesh)
io.full_print(fsp.omega_0, 'omega_0', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, lmsh.mesh)
io.full_print(fsp.rho_0, 'rho_0', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, lmsh.mesh))0, 'zeta_0', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, lmsh.mesh)

fsp.assigner.assign(fsp.phi, [fsp.psi_0_read, fsp.omega_0_read, fsp.rho_0_read, fsp.zeta_0_read])
######


bc_psi_l = DirichletBC(fsp.Q.sub(0), rpam.parameters['psi_l'], rmsh.boundary_l)
bc_psi_r = DirichletBC(fsp.Q.sub(0), rpam.parameters['psi_r'], rmsh.boundary_r)

bc_omega_l = DirichletBC(fsp.Q.sub(1), rpam.parameters['omega_l'], rmsh.boundary_l)

bc_rho_l = DirichletBC(fsp.Q.sub(2), rpam.parameters['rho_l'], rmsh.boundary_l)
bc_zeta_l = DirichletBC(fsp.Q.sub(3), rpam.parameters['zeta_l'], rmsh.boundary_l)

bcs = [bc_psi_l, bc_psi_r, bc_omega_l, bc_rho_l, bc_zeta_l]

# Define variational problem

F_psi = ( \
                    ( \
                                rpam.parameters['kappa'] * ( \
                                    1.0 / (8.0 * (fsp.rho) ** 3) * (5.0 * sin(fsp.psi) + sin(3 * fsp.psi)) - \
                                    1.0 / (4.0 * (fsp.rho) ** 2) * fsp.omega * (1.0 + 3.0 * cos(2.0 * fsp.psi)) - \
                                    3.0 / (2.0 * fsp.rho) * sin(fsp.psi) * ((fsp.omega) ** 2) + \
                                    1.0 / 2.0 * (fsp.omega) ** 3 + \
                                    2.0 / (fsp.rho) * cos(fsp.psi) * fsp.omega.dx(0) \
                            ) - \
                                fsp.sigma * (1.0 / (fsp.rho) * sin(fsp.psi) + fsp.omega) \
                        ) * fsp.nu_psi - \
                    rpam.parameters['kappa'] * fsp.omega.dx(0) * fsp.nu_psi.dx(0)
        ) * rmsh.dx + \
        (rpam.parameters['kappa'] * bgeo.facet_normal[0] * fsp.omega.dx(0) * fsp.nu_psi) * rmsh.ds
'''
omega = \partial_1 psi
<omega nu_omega>_Omega = <(\partial_1 psi) nu_omega>_Omega
                       = <(\partial_i psi) \delta_{i1} nu_omega>_Omega
                       = - <psi \partial_1 nu_omega>_Omega + <n_1 psi nu_omega>_{\partial Omega}
                       
<omega nu_omega + psi \partial_1 nu_omega>_Omega - <n_1 psi nu_omega>_{\partial Omega} = 0                
'''
F_omega = (fsp.omega * fsp.nu_omega + fsp.psi * fsp.nu_omega.dx(0)) * rmsh.dx - \
          (bgeo.facet_normal[0] * fsp.psi * fsp.nu_omega) * rmsh.ds

'''
\partial_1 rho = cos(psi)
<(\partial_1 rho) nu_rho >_Omega = <cos(psi) nu_rho>_Omega
<(\partial_i rho)  \delta_{i1} nu_rho >_Omega = <cos(psi) nu_rho>_Omega
-< rho  \partial_1 nu_rho >_Omega + <n_1  rho nu_rho >_{\partial Omega} = <cos(psi) nu_rho>_Omega 

<cos(psi) nu_rho + rho  \partial_1 nu_rho >_Omega  - <n_1  rho nu_rho>_{\partial Omega} = 0
'''

F_rho = (cos(fsp.psi) * fsp.nu_rho + fsp.rho * fsp.nu_rho.dx(0)) * rmsh.dx - \
        (bgeo.facet_normal[0] * fsp.rho * fsp.nu_rho) * rmsh.ds

F_zeta = (-sin(fsp.psi) * fsp.nu_zeta + fsp.zeta * fsp.nu_zeta.dx(0)) * rmsh.dx - \
         (bgeo.facet_normal[0] * fsp.zeta * fsp.nu_zeta) * rmsh.ds

'''
F_N = rpam.parameters["alpha"] / rmsh.r_mesh * ( \
        + ( (fsp.omega - fsp.psi.dx(0)) * fsp.nu_omega) * rmsh.ds + \
        + (( fsp.rho.dx(0) - cos(fsp.psi)) * fsp.nu_rho) * rmsh.ds + \
        + (( fsp.zeta.dx(0) + sin(fsp.psi)) * fsp.nu_zeta) * rmsh.ds  \
    )
'''

# total functional for the mixed problem
F = (F_psi + F_omega + F_rho + F_zeta)
