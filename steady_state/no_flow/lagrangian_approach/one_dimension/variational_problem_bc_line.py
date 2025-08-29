from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import command as cmd
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import read_parameters_solve as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

cmd.set_gauge('arc_length')

i, j, k, l = ufl.indices(4)


class sigma_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters["sigma_const"]

    def value_shape(self):
        return (1,)


fsp.sigma.interpolate(sigma_Expression(element=fsp.Q_sigma.ufl_element()))

'''
fsp.psi_exact.interpolate(z_exact_Expression(element=fsp.Q_psi.ufl_element()))
fsp.X_exact.interpolate(omega_exact_Expression(element=fsp.Q_X.ufl_element()))
fsp.mu_exact.interpolate(mu_exact_Expression(element=fsp.Q_mu.ufl_element()))
'''
# uncomment this to set the initial profiles from the ODE soltion
'''
print("Reading the initial profiles from file ...")
fu.set_from_file( fsp.z_0_read, 'solution-ode/z_ode.csv' )
fsp.z_0.interpolate( z_0_Expression( element=fsp.Q_z.ufl_element() ) )

fu.set_from_file( fsp.omega_0_r_read, 'solution-ode/omega_ode.csv' )
fsp.omega_0.interpolate( omega_0_Expression( element=fsp.Q_omega.ufl_element() ) )

fu.set_from_file( fsp.mu_0_read, 'solution-ode/mu_ode.csv' )
fsp.mu_0.interpolate( mu_0_Expression( element=fsp.Q_mu.ufl_element() ))

fsp.tau_exact.interpolate( tau_exact_Expression( element=fsp.Q_tau.ufl_element() ) )

#uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
fsp.assigner.assign(fsp.psi, [fsp.z_0, fsp.omega_0, fsp.mu_0])
print("... done")
'''

# boundary conditions (BCs)

bc_psi_l = DirichletBC(fsp.Q.sub(0), Constant(rpam.parameters["psi_l"]), rmsh.boundary_l)
bc_psi_r = DirichletBC(fsp.Q.sub(0), Constant(rpam.parameters["psi_r"]), rmsh.boundary_r)
bc_mu_l = DirichletBC(fsp.Q.sub(1), Constant(rpam.parameters["mu_l"]), rmsh.boundary_l)
bc_X_l = DirichletBC(fsp.Q.sub(2), Constant((rpam.parameters["X_l"][0], rpam.parameters["X_l"][1])), rmsh.boundary_l)

bcs = [bc_psi_l, bc_psi_r, bc_mu_l, bc_X_l]

# Define variational problem

'''
F_psi = - 4 * rpam.parameters['kappa'] *  fsp.mu.dx(0).dx(0)  * fsp.nu_psi * rmsh.dx = 
=  4 * rpam.parameters['kappa'] *  fsp.mu.dx(0)  * fsp.nu_psi.dx(0) * rmsh.dx 
- 4 * rpam.parameters['kappa'] *  fsp.mu.dx(0)  * fsp.nu_psi * bgeo.facet_normal[0] * rmsh.ds

'''

F_psi = ( \
                    rpam.parameters["kappa"] * ( \
                        geo.g_c(fsp.psi)[i, j] * (fsp.mu.dx(j)) * (fsp.nu_psi.dx(i)) - 2.0 * fsp.mu * ((fsp.mu ** 2) - geo.K(fsp.psi)) * fsp.nu_psi) + fsp.sigma * fsp.mu * fsp.nu_psi \
            ) * geo.sqrt_detg(fsp.psi) * rmsh.dx \
        - ( \
                    (rpam.parameters["kappa"] * (bgeo.n_lr(fsp.psi))[i] * fsp.nu_psi * (fsp.mu.dx(i))) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds_l + \
                    (rpam.parameters["kappa"] * (bgeo.n_lr(fsp.psi))[i] * fsp.nu_psi * (fsp.mu.dx(i))) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds_r
        )

F_mu = ((geo.H(fsp.psi) - fsp.mu) * fsp.nu_mu) * geo.sqrt_detg(fsp.psi) * rmsh.dx

F_X = (fsp.X[i].dx(0) - geo.e(fsp.psi)[0, i]) * fsp.nu_X[i] * rmsh.dx

F_N = rpam.parameters["alpha"] / rmsh.r_mesh * ( \
    # these terms constrain mu = H(psi) on the boundary
        ((geo.H(fsp.psi) - fsp.mu) * fsp.nu_mu) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds_l + \
        ((geo.H(fsp.psi) - fsp.mu) * fsp.nu_mu) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds_r \
    )

# total functional for the mixed problem
F = (F_psi + F_mu + F_X) + F_N
