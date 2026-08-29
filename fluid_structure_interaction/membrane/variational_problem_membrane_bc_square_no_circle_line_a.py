from fenics import *
import importlib
import numpy as np
import ufl as ufl


import command as cmd
import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import differential_geometry.manifold.gauges.arc_length_gauge as geo_al
import physics.fluid_mechanics as flu
import function_spaces as fsp
import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

cmd.set_gauge('arc_length')


i, j, k, l, alpha, beta = ufl.indices( 6 )


dt = rpam.parameters['T'] / rpam.parameters['N']

# reference configuration of the manifold, a straight line which coincides with the mesh line
class X_ref_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = x[0]
        values[1] = rmsh.parameters['h']

    def value_shape(self):
        return (2,)

# expressions for the initial conditions
class v_n_0_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters['v_bar_l'][0]

    def value_shape(self):
        return (1,)

class sigma_n_32_0_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters['sigma_n_12_0']

    def value_shape(self):
        return (1,)

class nu_n_12_0_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 1

    def value_shape(self):
        return (1,)
    
class U_n_12_0_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)
    
    
# expressions for the boundary conditions
class v_bar_l_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters['v_bar_l'][0]

    def value_shape(self):
        return (1,)
    
class v_bar_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = rpam.parameters['v_bar_r'][0]

    def value_shape(self):
        return (1,)
        
        
        
fsp.X_ref.interpolate(X_ref_Expression(element=fsp.Q_X.ufl_element()))

fsp.v_bar_l.interpolate( v_bar_l_Expression( element=fsp.Q_v_bar.ufl_element() ) )
fsp.v_bar_r.interpolate( v_bar_r_Expression( element=fsp.Q_v_bar.ufl_element() ) )



# boundary conditions

# bc_v_bar_l = DirichletBC(fsp.Q_mem.sub(0), fsp.v_bar_l, rmsh.lmsh.mf_sub_meshes[1], rmsh.parameters['vertex_sub_mesh_1_l_id'])
bc_v_bar_r = DirichletBC(fsp.Q_mem.sub(0), fsp.v_bar_r, rmsh.lmsh.mf_sub_meshes[1], rmsh.parameters['vertex_sub_mesh_1_r_id'])

bc_w_bar_l = DirichletBC(fsp.Q_mem.sub(1), Constant(0), rmsh.lmsh.mf_sub_meshes[1], rmsh.parameters['vertex_sub_mesh_1_l_id'])

bc_phi_l = DirichletBC(fsp.Q_mem.sub(2), Constant(0), rmsh.lmsh.mf_sub_meshes[1], rmsh.parameters['vertex_sub_mesh_1_l_id'])

bc_U_n_12_l = DirichletBC(fsp.Q_mem.sub(5), Constant((0,0)), rmsh.lmsh.mf_sub_meshes[1], rmsh.parameters['vertex_sub_mesh_1_l_id'])
bc_U_n_12_0_r = DirichletBC(fsp.Q_mem.sub(5).sub(0), Constant(0), rmsh.lmsh.mf_sub_meshes[1], rmsh.parameters['vertex_sub_mesh_1_r_id'])







#BCs
bcs_mem = [bc_v_bar_r, bc_w_bar_l, bc_phi_l, bc_U_n_12_l, bc_U_n_12_0_r]



# Define variational problem : F_vbar, F_wbar .... F_mu_n_12 are related to the PDEs for v_bar, ..., mu^{n-1/2} respectively .
# natural BC imposed here
F_v_bar = ( \
                      rpam.parameters['rho'] * (( \
                                         (fsp.v_bar[i] - fsp.v_n_1[i]) \
                                         + dt * ((3.0 / 2.0 * fsp.v_n_1[j] - 1.0 / 2.0 * fsp.v_n_2[j]) * geo.Nabla_v( fsp.V, fsp.psi_n_12, fsp.nu_n_12 )[i, j] \
                                                     - 2.0 * fsp.V[j] * fsp.W * geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, k] * geo.b( fsp.psi_n_12, fsp.nu_n_12 )[k, j]) \

                                 ) * fsp.nu_v_bar[i] \
                             + dt * 1.0 / 2.0 * (fsp.W ** 2) * geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, j] * geo.Nabla_f( fsp.nu_v_bar, fsp.psi_n_12, fsp.nu_n_12 )[i, j] \
                             ) \
                      + dt * (fsp.sigma_n_32 * geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, j] * geo.Nabla_f( fsp.nu_v_bar, fsp.psi_n_12, fsp.nu_n_12 )[i, j] \
                                  + 2.0 * rpam.parameters['eta'] * geo.d_c( fsp.V, fsp.W, fsp.psi_n_12, fsp.nu_n_12 )[i, j] * geo.Nabla_f( fsp.nu_v_bar, fsp.psi_n_12, fsp.nu_n_12 )[j, i] \
                                    #   force exerted by the fluid on the membrane
                                      -  geo.from_3D_to_tangent(fsp.psi_n_12, 
                                                             flu.dFdl(
                                                                 fsp.var_tensor_sigma_fl_on_mem, 
                                                                 geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)
                                                                 ), 
                                                             fsp.nu_n_12)[i] * fsp.nu_v_bar[i]\
                            )
          ) * geo.sqrt_detg( fsp.psi_n_12, fsp.nu_n_12 ) * rmsh.dx_sub_mesh[1]  \
          - dt * rpam.parameters['rho'] / 2.0 * ( \
                      ((fsp.W ** 2) * (bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12,  lmsh.sub_meshes[1]))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds_sub_mesh[1]['ds'] \
          ) \
          - dt * ( \
                      (fsp.sigma_n_32 * (bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12,  lmsh.sub_meshes[1]))[i] * fsp.nu_v_bar[i]) * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds_sub_mesh[1]['ds'] \
           ) \
          - dt * 2.0 * rpam.parameters['eta'] * ( \
                      (geo.d_c( fsp.V, fsp.W, fsp.psi_n_12, fsp.nu_n_12 )[i, j] * geo.g( fsp.psi_n_12, fsp.nu_n_12 )[i, k] * (bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12,  lmsh.sub_meshes[1]))[k] * fsp.nu_v_bar[j]) * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds_sub_mesh[1]['ds_r']
          )


F_w_bar = ( \
                      rpam.parameters['rho'] * ((fsp.w_bar - fsp.w_n_1) + dt * fsp.V[i] * fsp.V[k] * geo.b( fsp.psi_n_12, fsp.nu_n_12 )[k, i]) * fsp.nu_w_bar \
                      - dt * rpam.parameters['rho'] * fsp.W * geo.Nabla_v( geo.vector_times_scalar( 3.0 / 2.0 * fsp.v_n_1 - 1.0 / 2.0 * fsp.v_n_2, fsp.nu_w_bar ), fsp.psi_n_12, fsp.nu_n_12 )[i, i] \
                      + dt * 2.0 * rpam.parameters['kappa'] * ( \
                                  - geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, j] * ((fsp.mu_n_12).dx( j )) * (fsp.nu_w_bar.dx( i )) \
                                  + 2.0 * fsp.mu_n_12 * (((fsp.mu_n_12) ** 2) - geo.K( fsp.psi_n_12, fsp.nu_n_12 )) * fsp.nu_w_bar \
                          ) \
                      - dt * ( \
                                  2.0 * fsp.sigma_n_32 * fsp.mu_n_12 \
                                  + 2.0 * rpam.parameters['eta'] * (geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, k] * geo.Nabla_v( fsp.V, fsp.psi_n_12, fsp.nu_n_12 )[j, k] *
                                                 (geo.b( fsp.psi_n_12, fsp.nu_n_12 ))[i, j] - 2.0 * fsp.W * (
                                                         2.0 * ((fsp.mu_n_12) ** 2) - geo.K( fsp.psi_n_12, fsp.nu_n_12 )))\
                                    #   force exerted by the fluid on the membrane
                                  + geo.from_3D_to_normal(fsp.psi_n_12, 
                                                          flu.dFdl(
                                                                 fsp.var_tensor_sigma_fl_on_mem, 
                                                                 geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)
                                                                 ), 
                                                          fsp.nu_n_12)
                                                             
                      ) * fsp.nu_w_bar
          ) * geo.sqrt_detg( fsp.psi_n_12, fsp.nu_n_12 ) * rmsh.dx_sub_mesh[1] \
          + dt * rpam.parameters['rho'] * ( \
                      (fsp.W * fsp.nu_w_bar * (bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12,  lmsh.sub_meshes[1]))[j] * geo.g( fsp.psi_n_12, fsp.nu_n_12 )[j, i] * (3.0 / 2.0 * fsp.v_n_1[i] - 1.0 / 2.0 * fsp.v_n_2[i])) * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds_sub_mesh[1]['ds'] \

          ) \
          + dt * 2.0 * rpam.parameters['kappa'] * ( \
                      (fsp.nu_w_bar * (bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12,  lmsh.sub_meshes[1]))[i] * ((fsp.mu_n_12).dx( i ))) * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds_sub_mesh[1]['ds'] \
          )
          

          

# natural BC implemented here
F_phi = ( \
                    dt * geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, j] * (fsp.phi.dx( i )) * (fsp.nu_phi.dx( j )) \
                    + rpam.parameters['rho'] * (geo.Nabla_v( fsp.v_bar, fsp.psi_n_12, fsp.nu_n_12 )[i, i] - 2.0 * fsp.mu_n_12 * fsp.w_bar) * fsp.nu_phi \
            ) * geo.sqrt_detg( fsp.psi_n_12, fsp.nu_n_12 ) * rmsh.dx_sub_mesh[1] \
        - ((bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12,  lmsh.sub_meshes[1]))[i] * (fsp.phi).dx(i)) * fsp.nu_phi * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds_sub_mesh[1]['ds_l']





F_v_n = ((rpam.parameters['rho'] * (fsp.v_n[i] - fsp.v_bar[i]) + dt * geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, j] * (fsp.phi.dx( j ))) * fsp.nu_v_n[i]) * geo.sqrt_detg( fsp.psi_n_12, fsp.nu_n_12 ) * rmsh.dx_sub_mesh[1]




F_w_n = ((fsp.w_n - fsp.w_bar) * fsp.nu_w_n) * geo.sqrt_detg( fsp.psi_n_12, fsp.nu_n_12 ) * rmsh.dx_sub_mesh[1]




F_U_n_12 = ( \
                    ( \
                                (fsp.U_n_12[alpha] - fsp.U_n_32[alpha]) \
                                - dt * fsp.w_n_1 * (geo.normal( fsp.psi_n_12, fsp.nu_n_12 ))[alpha]  \
                        ) * fsp.nu_U_n_12[alpha] \
            ) * geo.sqrt_detg( fsp.psi_n_12, fsp.nu_n_12 ) * rmsh.dx_sub_mesh[1]



F_nu_psi = (
        ((fsp.X_ref[0] + fsp.U_n_12[0]).dx(0) - geo.e(fsp.psi_n_12, fsp.nu_n_12)[0, 0])\
        * ( -cos(fsp.psi_n_12) * fsp.nu_nu_n_12 + fsp.nu_n_12 * sin(fsp.psi_n_12) * fsp.nu_psi_n_12 )\
        +  ((fsp.X_ref[1] + fsp.U_n_12[1]).dx(0) - geo.e(fsp.psi_n_12, fsp.nu_n_12)[0, 1])\
        * ( sin(fsp.psi_n_12) * fsp.nu_nu_n_12 + fsp.nu_n_12 * cos(fsp.psi_n_12) * fsp.nu_psi_n_12 )\
    ) * geo.sqrt_detg(fsp.psi_n_12, fsp.nu_n_12) * rmsh.dx_sub_mesh[1]


F_mu_n_12 = ((geo.H( fsp.psi_n_12, fsp.nu_n_12 ) - fsp.mu_n_12) * fsp.nu_mu_n_12) * geo.sqrt_detg( fsp.psi_n_12, fsp.nu_n_12 ) * rmsh.dx_sub_mesh[1]



F_N =  rpam.parameters["alpha"] / rmsh.r_mesh[1] * (
        # this term constrains mu_n_12 = H(omega_n_12) on the boundary
        ((geo.H(fsp.psi_n_12, fsp.nu_n_12) - fsp.mu_n_12) * fsp.nu_mu_n_12) * bgeo.sqrt_deth_lr(fsp.psi_n_12) * rmsh.ds_sub_mesh[1]['ds'] \
        + (\
              ((fsp.X_ref[0] + fsp.U_n_12[0]).dx(0) - geo.e(fsp.psi_n_12, fsp.nu_n_12)[0, 0]) * ( -cos(fsp.psi_n_12) * fsp.nu_nu_n_12 + fsp.nu_n_12 * sin(fsp.psi_n_12) * fsp.nu_psi_n_12 )\
              + ((fsp.X_ref[1] + fsp.U_n_12[1]).dx(0) - geo.e(fsp.psi_n_12, fsp.nu_n_12)[0, 1]) * ( sin(fsp.psi_n_12) * fsp.nu_nu_n_12 + fsp.nu_n_12 * cos(fsp.psi_n_12) * fsp.nu_psi_n_12 )\
        ) * bgeo.sqrt_deth_lr(fsp.psi_n_12) * rmsh.ds_sub_mesh[1]['ds']\
        + (\
        # this implements BC (79) 
            (fsp.w_bar.dx(i)) * geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, j] *(fsp.nu_w_bar.dx(j)) \
        # this implements BC (66)
            +   (fsp.U_n_12[1].dx(i)) * geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, j] * (fsp.nu_U_n_12[1].dx(j))
        ) * bgeo.sqrt_deth_lr(fsp.psi_n_12) * rmsh.ds_sub_mesh[1]['ds_r']    )

# total functional for the mixed problem
F_mem = (F_v_bar + F_w_bar + F_phi + F_v_n + F_w_n + F_U_n_12 + F_nu_psi + F_mu_n_12) + F_N

