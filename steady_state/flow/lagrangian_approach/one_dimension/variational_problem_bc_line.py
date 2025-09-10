'''
In this variational problem, we solve for a membrane profile at steady state with flows, by fixing the 'slope' \psi and the position X^\alpha at both ends of the membrane.

To achieve this, the stretch parameter \nu is considered as a variable, and it is solved for by imposing that \partial_1 \nu = 0:
this is achieved by considering a penalty term G = \alpha/h \int dx (\partial_1 \nu)^2 : the variation of G with respect to \nu is F_nu

The BC (51) in 'Lagrangian approach'  is replaced by the BC X1_r = X1_r_0, and, given that \nu is a constant field we can chose \nu by fixing an additional BC, which is
 X2_r = X2r_0.
'''

from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import command as cmd
import function_spaces as fsp
import function as fu
import differential_geometry.manifold.geometry as geo
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

cmd.set_gauge('arc_length')

i, j, k, l, alpha = ufl.indices(5)


# uncomment this to set the initial profiles from the ODE soltion
#
print("Reading the initial profiles from file ...")
fu.read_from_file('solution_ode/v.csv', fsp.v_0)
fu.read_from_file('solution_ode/w.csv', fsp.w_0)
fu.read_from_file('solution_ode/sigma.csv', fsp.sigma_0)
fu.read_from_file('solution_ode/psi.csv', fsp.psi_0)
fu.read_from_file('solution_ode/mu.csv', fsp.mu_0)
fu.read_from_file('solution_ode/X.csv', fsp.X_0)
fu.read_from_file('solution_ode/nu.csv', fsp.nu_0)

fsp.assigner.assign(fsp.phi, [fsp.v_0, fsp.w_0, fsp.sigma_0, fsp.psi_0, fsp.mu_0, fsp.X_0, fsp.nu_0 ])
print('... done')
#

# boundary conditions (BCs)
bc_v_l = DirichletBC(fsp.Q.sub(0), Constant((rpam.parameters['v_l'])), rmsh.boundary_l)
bc_v_r = DirichletBC(fsp.Q.sub(0), Constant((rpam.parameters['v_r'])), rmsh.boundary_r)

bc_w = DirichletBC(fsp.Q.sub(1), Constant(rpam.parameters['w_lr']), rmsh.boundary_l)
bc_sigma_r = DirichletBC(fsp.Q.sub(2), Constant(rpam.parameters['sigma_r']), rmsh.boundary_r)

bc_psi_l = DirichletBC(fsp.Q.sub(3), Constant(rpam.parameters["psi_l"]), rmsh.boundary_l)
bc_psi_r = DirichletBC(fsp.Q.sub(3), Constant(rpam.parameters["psi_r"]), rmsh.boundary_r)

bc_X_l = DirichletBC(fsp.Q.sub(5), Constant((rpam.parameters["X_l"][0], rpam.parameters["X_l"][1])), rmsh.boundary_l)
bc_X_r = DirichletBC(fsp.Q.sub(5), Constant((rpam.parameters["X_r"][0], rpam.parameters["X_r"][1])), rmsh.boundary_r)

bcs = [bc_v_l, bc_v_r, bc_w, bc_sigma_r, bc_psi_l, bc_psi_r, bc_X_l, bc_X_r]

# variational problem


F_v = ( \
                  rpam.parameters['rho'] * ( \
                      (fsp.v[j] * geo.Nabla_v(fsp.v, fsp.psi, fsp.nu)[i, j] - 2.0 * fsp.v[j] * fsp.w * geo.g_c(fsp.psi, fsp.nu)[i, k] * geo.b(fsp.psi, fsp.nu)[k, j]) * fsp.nu_v[i] \
                      + 1.0 / 2.0 * (fsp.w ** 2) * geo.g_c(fsp.psi, fsp.nu)[i, j] * geo.Nabla_f(fsp.nu_v, fsp.psi, fsp.nu)[i, j] \
              ) \
                  + (fsp.sigma * geo.g_c(fsp.psi, fsp.nu)[i, j] * geo.Nabla_f(fsp.nu_v, fsp.psi, fsp.nu)[i, j] \
                     + 2.0 * rpam.parameters['eta'] * geo.d_c(fsp.v, fsp.w, fsp.psi, fsp.nu)[j, i] * geo.Nabla_f(fsp.nu_v, fsp.psi, fsp.nu)[j, i])
      ) * geo.sqrt_detg(fsp.psi, fsp.nu) * rmsh.dx \
      - rpam.parameters['rho'] / 2.0 * ( \
                  ((fsp.w ** 2) * (bgeo.n_lr(fsp.psi, fsp.nu))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
          ) \
      - ( \
                  (fsp.sigma * (bgeo.n_lr(fsp.psi, fsp.nu))[i] * fsp.nu_v[i]) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
          ) \
      - 2.0 * rpam.parameters['eta'] * ( \
                  (geo.d_c(fsp.v, fsp.w, fsp.psi, fsp.nu)[i, j] * geo.g(fsp.psi, fsp.nu)[i, k] * (bgeo.n_lr(fsp.psi, fsp.nu))[k] * fsp.nu_v[j]) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
          )

F_w = (fsp.w * fsp.nu_w) * geo.sqrt_detg(fsp.psi, fsp.nu) * rmsh.dx

F_sigma = (geo.Nabla_v(fsp.v, fsp.psi, fsp.nu)[i, i] - 2.0 * fsp.mu * fsp.w) * fsp.nu_sigma * geo.sqrt_detg(fsp.psi, fsp.nu) * rmsh.dx

F_psi = ( \
                    rpam.parameters['rho'] * (fsp.v[i] * fsp.v[k] * geo.b(fsp.psi, fsp.nu)[k, i]) * fsp.nu_psi \
                    - rpam.parameters['rho'] * fsp.w * geo.Nabla_v(geo.vector_times_scalar(fsp.v, fsp.nu_psi), fsp.psi, fsp.nu)[i, i] \
                    + 2.0 * rpam.parameters['kappa'] * ( \
                                - geo.g_c(fsp.psi, fsp.nu)[i, j] * (fsp.mu.dx(i)) * (fsp.nu_psi.dx(j)) \
                                + 2.0 * fsp.mu * (fsp.mu ** 2 - geo.K(fsp.psi, fsp.nu)) * fsp.nu_psi \
                        ) \
                    - ( \
                                2.0 * fsp.sigma * fsp.mu \
                                + 2.0 * rpam.parameters['eta'] * (geo.g_c(fsp.psi, fsp.nu)[i, k] * geo.Nabla_v(fsp.v, fsp.psi, fsp.nu)[j, k] *
                                                                  (geo.b(fsp.psi, fsp.nu))[i, j] - 2.0 * fsp.w * (2.0 * fsp.mu ** 2 - geo.K(fsp.psi, fsp.nu)))
                    ) * fsp.nu_psi
        ) * geo.sqrt_detg(fsp.psi, fsp.nu) * rmsh.dx \
        + rpam.parameters['rho'] * ( \
                    (fsp.w * fsp.nu_psi * (bgeo.n_lr(fsp.psi, fsp.nu))[j] * geo.g(fsp.psi, fsp.nu)[j, i] * fsp.v[i]) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
            ) \
        + 2.0 * rpam.parameters['kappa'] * ( \
                    ((bgeo.n_lr(fsp.psi, fsp.nu))[i] * (fsp.mu.dx(i)) * fsp.nu_psi) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
            )

F_mu = ((fsp.mu - geo.H(fsp.psi, fsp.nu)) * fsp.nu_mu) * geo.sqrt_detg(fsp.psi, fsp.nu) * rmsh.dx

F_X = (fsp.X[alpha].dx(0) - geo.e(fsp.psi, fsp.nu)[0, alpha]) * fsp.nu_X[alpha] * geo.sqrt_detg(fsp.psi, fsp.nu) * rmsh.dx

F_nu = rpam.parameters["alpha"] / rmsh.r_mesh * (fsp.nu.dx(i) * fsp.nu_nu.dx(i)) * rmsh.dx

F_N = rpam.parameters["alpha"] / rmsh.r_mesh * ( \
    # these terms constrain mu = H(psi) on the boundary
        ((fsp.mu - geo.H(fsp.psi, fsp.nu)) * fsp.nu_mu) * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
        + (fsp.X[alpha].dx(0) - geo.e(fsp.psi, fsp.nu)[0, alpha]) * fsp.nu_X[alpha] * bgeo.sqrt_deth_lr(fsp.psi) * rmsh.ds \
     )

# total functional for the mixed problem
F = (F_v + F_w + F_sigma + F_psi + F_mu + F_X + F_nu) + F_N
