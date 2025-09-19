from fenics import *
import importlib
import ufl as ufl

import command as cmd
import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import function as fu
import input_output as io
import parameters.read.solution as rpam
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

cmd.set_gauge('arc_length')

i, j, k, l, alpha = ufl.indices(5)


class nu_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = (4+x[0]**4)/(1.0+x[0]**2)

    def value_shape(self):
        return (1,)


class sigma_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters["sigma_const"]

    def value_shape(self):
        return (1,)
    
# reference configuration of the manifold, a straight line which coincides with the mesh line
class X_r_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = x[0]
        values[1] = 0

    def value_shape(self):
        return (2,)




fsp.sigma.interpolate(sigma_Expression(element=fsp.Q_sigma.ufl_element()))
fsp.nu.interpolate(nu_Expression(element=fsp.Q_nu.ufl_element()))
fsp.X_r.interpolate(X_r_Expression(element=fsp.Q_X.ufl_element()))


# uncomment this to set the initial profiles from the ODE soltion
#
print("Reading the initial profiles from file ...")
print(f'solution ode path = {rpam.parameters["solution_ode_path"]}')
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'psi.csv', fsp.psi_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'mu.csv', fsp.mu_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'u.csv', fsp.u_0)

fsp.assigner.assign(fsp.phi, [fsp.psi_0, fsp.mu_0, fsp.u_0])
print('... done')
# 


print("... done")

# boundary conditions (BCs)

bc_psi_l = DirichletBC(fsp.Q.sub(0), Constant(rpam.parameters["psi_l"]), rmsh.boundary_l)
bc_psi_r = DirichletBC(fsp.Q.sub(0), Constant(rpam.parameters["psi_r"]), rmsh.boundary_r)
bc_mu_l = DirichletBC(fsp.Q.sub(1), Constant(rpam.parameters["mu_l"]), rmsh.boundary_l)
bc_u_l = DirichletBC(fsp.Q.sub(2), Constant((rpam.parameters["u_l"][0], rpam.parameters["u_l"][1])), rmsh.boundary_l)

bcs = [bc_psi_l, bc_psi_r, bc_mu_l, bc_u_l]

# Define variational problem


F_psi = ( \
                    rpam.parameters["kappa"] * ( \
                        geo.g_c(fsp.psi, fsp.nu)[i, j] * (fsp.mu.dx(j)) * (fsp.nu_psi.dx(i)) \
                        - 2.0 * fsp.mu * ((fsp.mu ** 2) - geo.K(fsp.psi, fsp.nu)) * fsp.nu_psi) + fsp.sigma * fsp.mu * fsp.nu_psi \
            ) * geo.sqrt_detg(fsp.psi, fsp.nu) * rmsh.dx \
        - ( \
                    (rpam.parameters["kappa"] * (bgeo.n_lr(fsp.psi, fsp.nu))[i] * fsp.nu_psi * (fsp.mu.dx(i))) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds
        )

F_mu = ((fsp.mu - geo.H(fsp.psi, fsp.nu)) * fsp.nu_mu) * geo.sqrt_detg(fsp.psi, fsp.nu) * rmsh.dx

'''
u[alpha] +  X_r^alpha = X^alpha
'''

F_X = ((fsp.X_r[alpha] + fsp.u[alpha]).dx(0) - geo.e(fsp.psi, fsp.nu)[0, alpha]) * fsp.nu_u[alpha] * geo.sqrt_detg(fsp.psi, fsp.nu) * rmsh.dx

F_N = rpam.parameters["alpha"] / rmsh.r_mesh * ( \
    # these terms constrain mu = H(psi) on the boundary
        ((fsp.mu - geo.H(fsp.psi, fsp.nu)) * fsp.nu_mu) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
    )

# total functional for the mixed problem
F = (F_psi + F_mu + F_X) + F_N
