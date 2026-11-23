'''
this module solves for the fields, \textrm_{v_FL}^n, \varsigma,  which define the state of the fluid
'''

from fenics import *
import importlib
import ufl as ufl
import numpy as np

import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

alpha, beta, gamma, delta = ufl.indices(4)

dt = rpam.parameters['T'] / rpam.parameters['N']  # time step size


#1-order pressure field solution 
class sigma_fl_n_12_Expression(UserExpression):
    def eval(self, values, x):
        n=rmsh.parameters['n']
        q = n * np.pi
        A = rmsh.parameters['A']
        v = rpam.parameters['v_fl_bar_t_const']
        x=x[0]
        #analytical solutions of pressure up to 2nd order in membrane deformation
        values[0] =2*q**2*A*v*rpam.parameters['eta_fluid']*np.cos(q*x)-2*q**2*v*A**2*rpam.parameters['eta_fluid']*(np.sin(q*x))**2
    def value_shape(self):
        return (1,)

class v_fl_bar_b_Expression(UserExpression):
    def eval(self, values, x):
        n=rmsh.parameters['n']
        q = n * np.pi
        A = rmsh.parameters['A']
        v = 0.01
        x=x[0]
        #analytical solutions of the velocity up to second order in membrane deformation 
        values[0]= q*A*v*np.sin(q*x)
        values[1]= -v/4*(4-A**2*q**2+2*q**2*A**2*np.cos(2*q*x))
    def value_shape(self):
        return (2,)


fsp.v_fl_bar_b.interpolate(v_fl_bar_b_Expression(element=fsp.Q_v_fl_bar.ufl_element()))
fsp.sigma_fl_n_12.interpolate(sigma_fl_n_12_Expression(element=fsp.Q_phi_fl.ufl_element()))


# # BCs
bc_v_fl_bar_t = DirichletBC(fsp.Q_v_fl_bar,fsp.v_fl_bar_b,rmsh.lmsh.mf_sub_meshes[0],rmsh.parameters["sub_mesh_1_id"])

bc_v_fl_bar_0_l = DirichletBC(fsp.Q_v_fl_bar.sub(0), Constant(0),rmsh.lmsh.mf_sub_meshes[0], rmsh.parameters["line_sub_mesh_0_l_id"])
bc_v_fl_bar_0_r = DirichletBC(fsp.Q_v_fl_bar.sub(0), Constant(0),rmsh.lmsh.mf_sub_meshes[0],rmsh.parameters["line_sub_mesh_0_r_id"])

bc_v_fl_bar = [bc_v_fl_bar_0_l, bc_v_fl_bar_0_r, bc_v_fl_bar_t]

# 2) for step 2
bc_sigma_fl_t = DirichletBC(fsp.Q_phi_fl, fsp.sigma_fl_n_12,rmsh.lmsh.mf_sub_meshes[0],rmsh.parameters["sub_mesh_1_id"])
bc_sigma_fl = [bc_sigma_fl_t]

# step 1 for v_fl_bar
F_v_fl_bar = ( \
                   rpam.parameters['rho_fluid'] * (
                                                (fsp.v_fl_bar[alpha] - fsp.v_fl_n_1[alpha]) / dt \
                                                + (3.0 / 2.0 * (fsp.v_fl_n_1[gamma] - fsp.u_dot_n_1[gamma]) * ela.G(fsp.u_n_1)[beta, gamma] - 1.0 / 2.0 * (fsp.v_fl_n_2[gamma] - fsp.u_dot_n_2[gamma]) * ela.G(fsp.u_n_2)[beta, gamma]) * (fsp.V_fl[alpha]).dx(beta)
                                                ) * fsp.nu_v_fl_bar[alpha] \
                    + fsp.sigma_fl_n_32 * ela.G(fsp.u_n_1)[beta, alpha] * (fsp.nu_v_fl_bar[alpha]).dx(beta) \
                    + rpam.parameters['eta_fluid'] * ela.G(fsp.u_n_1)[gamma, beta] * ((fsp.V_fl[alpha]).dx(gamma)) * ela.G(fsp.u_n_1)[delta, beta] * (fsp.nu_v_fl_bar[alpha]).dx(delta) \
            ) * ela.detF(fsp.u_n_1) * rmsh.dx_sub_mesh[0] \
            - (ela.G(fsp.u_n_1)[beta, alpha] * (bgeo.sub_mesh_facet_normal[0])[beta] * fsp.sigma_fl_n_32 * fsp.nu_v_fl_bar[alpha]) * ela.detF(fsp.u_n_1) * rmsh.ds_sub_mesh[0]['ds_lr'] \
            - (ela.G(fsp.u_n_1)[beta, alpha] * (bgeo.sub_mesh_facet_normal[0])[beta] * fsp.sigma_fl_n_32 * fsp.nu_v_fl_bar[alpha]) * ela.detF(fsp.u_n_1) * rmsh.ds_sub_mesh[0]['ds_t'] \
            - ( \
                #natural boundary conditions imposed here
            rpam.parameters['eta_fluid'] * ela.G(fsp.u_n_1)[delta, beta] * (bgeo.sub_mesh_facet_normal[0])[delta] * ela.G(fsp.u_n_1)[gamma, beta] * (fsp.V_fl[alpha].dx(gamma)) * fsp.nu_v_fl_bar[alpha] \
             * ela.detF(fsp.u_n_1) * rmsh.ds_sub_mesh[0]['ds_lr'] +
             rpam.parameters['eta_fluid'] * ela.G(fsp.u_n_1)[delta, beta] * (bgeo.sub_mesh_facet_normal[0])[delta] * ela.G(fsp.u_n_1)[gamma, beta] * (fsp.V_fl[alpha].dx(gamma)) * fsp.nu_v_fl_bar[alpha] \
                          * ela.detF(fsp.u_n_1) * rmsh.ds_sub_mesh[0]['ds_t']               
                )+rpam.parameters["alpha"] / rmsh.r_mesh[0] * ( (fsp.v_fl_bar[1]).dx(0) * fsp.nu_v_fl_bar[1].dx(0) ) * rmsh.ds_sub_mesh[0]['ds_lr']


# step 2 for phi
F_phi = ( -dt*ela.G(fsp.u_n_1)[beta, alpha] * (fsp.phi_fl.dx(beta)) * ela.G(fsp.u_n_1)[delta, alpha] * (fsp.nu_phi_fl.dx(delta)) \
        - (rpam.parameters['rho_fluid'] ) * (ela.G(fsp.u_n_1)[beta, alpha] * (fsp.v_fl_bar[alpha]).dx(beta)) * fsp.nu_phi_fl \
        ) * ela.detF(fsp.u_n_1) * rmsh.dx_sub_mesh[0]  + \
        dt*(ela.G(fsp.u_n_1)[delta,  1] * (bgeo.sub_mesh_facet_normal[0])[delta] * ela.G(fsp.u_n_1)[beta, 1] * (fsp.phi_fl.dx(beta)) * fsp.nu_phi_fl) * ela.detF(fsp.u_n_1) * rmsh.ds_sub_mesh[0]['ds_lr']+\
        dt*(ela.G(fsp.u_n_1)[delta,  alpha] * (bgeo.sub_mesh_facet_normal[0])[delta] * ela.G(fsp.u_n_1)[beta, alpha] * (fsp.phi_fl.dx(beta)) * fsp.nu_phi_fl) * ela.detF(fsp.u_n_1) * rmsh.ds_sub_mesh[0]['ds_b']



F_phi_N = rpam.parameters["alpha"] / rmsh.r_mesh[0] * ( (fsp.phi_fl).dx(0) * fsp.nu_phi_fl.dx(0) ) * rmsh.ds_sub_mesh[0]['ds_l'] 

F_phi_fl=F_phi 

# step 3 for v_fl_n
F_v_fl_n = (((fsp.v_fl_bar[alpha] - fsp.v_fl_n[alpha]) - (dt / rpam.parameters['rho_fluid']) * ela.G(fsp.u_n_1)[beta, alpha] * (fsp.phi_fl.dx(beta))) * fsp.nu_v_fl_n[alpha]) * ela.detF(fsp.u_n_1) * rmsh.dx_sub_mesh[0] 






