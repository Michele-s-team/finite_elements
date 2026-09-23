'''
this module solves for the fields, \textrm_{v_FL}^n, \varsigma,  which define the state of the fluid
'''

from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import phi_lb as phi_lb
import physics.elasticity as ela
import physics.fluid_mechanics as flu
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

alpha, beta, gamma, delta = ufl.indices(4)

dt = rpam.parameters['T'] / rpam.parameters['N']  # time step size


# expressions for boundary conditions
class v_fl_bar_b_Expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = rpam.parameters['v_fl_bar_b_const']* 4.0 * 1.5 * x[0]/2 * (rmsh.parameters['L'] - x[0]/2) / (rmsh.parameters['L']**2)

    def value_shape(self):
        return (2,)


fsp.v_fl_bar_b.interpolate(v_fl_bar_b_Expression(element=fsp.Q_v_fl_bar.ufl_element()))

# checked - start
# BCs
# 1) for step 1
bc_v_fl_bar_b = DirichletBC(fsp.Q_v_fl_bar, fsp.v_fl_bar_b, rmsh.mf[0], rmsh.parameters["line_b_id"])
bc_v_fl_bar_l = DirichletBC(fsp.Q_v_fl_bar, Constant((0, 0)), rmsh.mf[0], rmsh.parameters["line_l_id"])
bc_v_fl_bar_0_r = DirichletBC(fsp.Q_v_fl_bar.sub(0), Constant(0), rmsh.mf[0], rmsh.parameters["line_r_id"])
bc_v_fl_bar_t = DirichletBC(fsp.Q_v_fl_bar, fsp.u_dot_n_1, rmsh.mf[0], rmsh.parameters["mesh_1_id"])

bc_v_fl_bar = [bc_v_fl_bar_b, bc_v_fl_bar_l, bc_v_fl_bar_0_r, bc_v_fl_bar_t]

# checked - end

# 2) for step 2
'''
BC that pins phi_fl to vertex tagged with rmsh.parameters["vertex_lb_id"]
this BC sets phi_fl(vertex_lb) = 0 and it removes the degeneracy in the variational problem 
'''
# coordinates of the bottom-left point of the mesh
x_lb = (rmsh.lmsh.mesh[0].coordinates()[rmsh.vf[0].array() ==  rmsh.parameters["vertex_lb_id"]])[0]
vertex_lb = CompiledSubDomain("near(x[0], x_lb_0) && near(x[1], x_lb_1)", x_lb_0=x_lb[0], x_lb_1=x_lb[1])
bc_phi_fl_vertex_lb = DirichletBC(fsp.Q_phi_fl, phi_lb.value, vertex_lb, method="pointwise")

bc_phi_fl = [bc_phi_fl_vertex_lb]


# step 1 for v_fl_bar
# checked - start
F_v_fl_bar_0 = ( \
                   rpam.parameters['rho_fluid'] * (
                                                (fsp.v_fl_bar[alpha] - fsp.v_fl_n_1[alpha]) / dt \
                                                + (3.0 / 2.0 * (fsp.v_fl_n_1[gamma] - fsp.u_dot_n_1[gamma]) * ela.G(fsp.u_n_1)[beta, gamma] - 1.0 / 2.0 * (fsp.v_fl_n_2[gamma] - fsp.u_dot_n_2[gamma]) * ela.G(fsp.u_n_2)[beta, gamma]) * (fsp.V_fl[alpha]).dx(beta)
                                                ) * fsp.nu_v_fl_bar[alpha] \
                    + ela.G(fsp.u_n_1)[gamma, beta] * flu.sigma_ale(fsp.V_fl, fsp.sigma_fl_n_32, fsp.u_n_1, rpam.parameters['eta_fluid'])[alpha, beta] * (fsp.nu_v_fl_bar[alpha]).dx(gamma) \
            ) * ela.detF(fsp.u_n_1) * rmsh.dx_mesh[0] \
            - (ela.G(fsp.u_n_1)[gamma, beta] * (bgeo.facet_normal[0])[gamma] * flu.sigma_ale(fsp.V_fl, fsp.sigma_fl_n_32, fsp.u_n_1, rpam.parameters['eta_fluid'])[alpha, beta] * fsp.nu_v_fl_bar[alpha]) * ela.detF(fsp.u_n_1) * rmsh.ds_mesh[0]['ds'] \

F_v_fl_bar_N = rpam.parameters["alpha"] / rmsh.r_mesh[0] * ( ela.G(fsp.u_n_1)[alpha, 0]* (fsp.V_fl[1].dx(alpha)) * ela.G(fsp.u_n_1)[beta, 0] * (fsp.nu_v_fl_bar[1].dx(beta)) ) * ela.detF(fsp.u_n_1) * rmsh.ds_mesh[0]['ds_r']    
        
F_v_fl_bar = F_v_fl_bar_0 + F_v_fl_bar_N
# checked - end

# step 2 for phi
# natural BC imposed here
F_phi_fl = ( \
                    - ela.G(fsp.u_n_1)[beta, alpha] * (fsp.phi_fl.dx(beta)) * ela.G(fsp.u_n_1)[delta, alpha] * (fsp.nu_phi_fl.dx(delta)) \
                    - (rpam.parameters['rho_fluid'] / dt) * ela.G(fsp.u_n_1)[beta, alpha] * ((fsp.v_fl_bar[alpha]).dx(beta)) * fsp.nu_phi_fl \
        ) * ela.detF(fsp.u_n_1) * rmsh.dx_mesh[0] + \
        (ela.G(fsp.u_n_1)[delta, 1] * (bgeo.facet_normal[0])[delta] * ela.G(fsp.u_n_1)[beta, 1] * (fsp.phi_fl.dx(beta)) * fsp.nu_phi_fl) * ela.detF(fsp.u_n_1) * rmsh.ds_mesh[0]['ds_r'] 



# step 3 for v_fl_n
F_v_fl_n = (((fsp.v_fl_bar[alpha] - fsp.v_fl_n[alpha]) - (dt / rpam.parameters['rho_fluid']) * ela.G(fsp.u_n_1)[beta, alpha] * (fsp.phi_fl.dx(beta))) * fsp.nu_v_fl_n[alpha]) * ela.detF(fsp.u_n_1) * rmsh.dx_mesh[0]


