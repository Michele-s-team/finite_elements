from fenics import *
import numpy as np
import ufl as ufl

#import function_spaces as fsp
#import boundary_geometry as bgeo
#import geometry as geo
#import read_mesh_ring as rmsh

'''
k : bending rigidity
z_r_const : value of z at the boundary \partial \Omega_r
z_R_const : value of z at the boundary \partial \Omega_R
omega_r_const : value of \partial_i z = omega_i at the boundary \partial \Omega_r
omega_R_const : value of \partial_i z = omega_i at the boundary \partial \Omega_R
'''
class vp:
    def __init__(self, geo, fsp, bgeo, rmsh):
        self.bgeo = bgeo
        self.geo = geo
        self.fsp = fsp
        self.rmsh = rmsh
        self.i, self.j, self.k, self.l = ufl.indices( 4 )
        # CHANGE PARAMETERS HERE
        self.kappa = 1.0
        C = 0.1
        z_r_const = 0.0
        z_R_const = 0.0
        omega_r_const = 0.0
        omega_R_const = 0.0
        # Nitche's parameter
        self.alpha = 1e2
        

        class SurfaceTensionExpression( UserExpression ):
            def eval(self, values, x):
                values[0] =  1.0
                # values[0] = ((2 + C**2) * kappa) / (2 * (1 + C**2) * geo.my_norm(x)**2)

            def value_shape(self):
                return (1,)


        class z_exact_Expression( UserExpression ):
            def eval(self, values, x):
                values[0] = C * geo.my_norm( x )

            def value_shape(self):
                return (1,)


        class omega_exact_Expression( UserExpression ):
            def eval(self, values, x):
                values[0] = C * x[0] / (geo.my_norm( x ))
                values[1] = C * x[1] / (geo.my_norm( x ))

            def value_shape(self):
                return (2,)


        class mu_exact_Expression( UserExpression ):
            def eval(self, values, x):
                values[0] = C / (2.0 * np.sqrt( 1.0 + C ** 2 ) * geo.my_norm( x ))

            def value_shape(self):
                return (1,)

        class nu_exact_Expression( UserExpression ):
            def eval(self, values, x):
                values[0] =  -((C * (1 + C**2) * (geo.my_norm(x))) / (2.0 * ((1 + C**2) * (geo.my_norm(x))**2)**(3.0/2.0))) * x[0]/geo.my_norm(x)
                values[1] = -((C * (1 + C**2) * (geo.my_norm(x))) / (2.0 * ((1 + C**2) * (geo.my_norm(x))**2)**(3.0/2.0))) * x[1]/geo.my_norm(x)

            def value_shape(self):
                return (2,)


        class tau_exact_Expression( UserExpression ):
            def eval(self, values, x):
                values[0] = C / (2.0 * ((1.0 + C ** 2) * (geo.my_norm( x )) ** 2) ** (3.0 / 2.0))

            def value_shape(self):
                return (1,)


        class z_r_Expression( UserExpression ):
            def eval(self, values, x):
                values[0] = z_r_const

            def value_shape(self):
                return (1,)


        class z_R_Expression( UserExpression ):
            def eval(self, values, x):
                values[0] = z_R_const

            def value_shape(self):
                return (1,)


        class omega_r_Expression( UserExpression ):
            def eval(self, values, x):
                values[0] = omega_r_const

            def value_shape(self):
                return (1,)


        class omega_R_Expression( UserExpression ):
            def eval(self, values, x):
                values[0] = omega_R_const

            def value_shape(self):
                return (1,)


        # CHANGE PARAMETERS HERE


        # values of z on ds_r and ds_R, to be used to check if the boundary conditions (BCs) are satisfied
        self.z_r = interpolate( z_r_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
        self.z_R = interpolate( z_R_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

        # values of \partial_i z = omega_i on the ds_r and ds_R, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
        self.omega_r = interpolate( omega_r_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )
        self.omega_R = interpolate( omega_R_Expression( element=fsp.Q_z.ufl_element() ), fsp.Q_z )

        fsp.sigma.interpolate( SurfaceTensionExpression( element=fsp.Q_sigma.ufl_element() ) )
        fsp.z_0.interpolate( z_exact_Expression( element=fsp.Q_z.ufl_element() ) )
        fsp.omega_0.interpolate( omega_exact_Expression( element=fsp.Q_omega.ufl_element() ) )
        fsp.mu_0.interpolate( mu_exact_Expression( element=fsp.Q_mu.ufl_element() ) )

        fsp.nu_0.interpolate( nu_exact_Expression( element=fsp.Q_nu.ufl_element() ) )
        fsp.tau_0.interpolate( tau_exact_Expression( element=fsp.Q_tau.ufl_element() ) )

        fsp.z_exact.interpolate( z_exact_Expression( element=fsp.Q_z.ufl_element() ) )
        fsp.omega_exact.interpolate( omega_exact_Expression( element=fsp.Q_omega.ufl_element() ) )
        fsp.mu_exact.interpolate( mu_exact_Expression( element=fsp.Q_mu.ufl_element() ) )

        fsp.nu_exact.interpolate( nu_exact_Expression( element=fsp.Q_nu.ufl_element() ) )
        fsp.tau_exact.interpolate( tau_exact_Expression( element=fsp.Q_tau.ufl_element() ) )

        # uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
        # fsp.assigner.assign(fsp.psi, [fsp.z_0, fsp.omega_0, fsp.mu_0])

    def change_dirichlet_bc(self, z_r_const, z_R_const):
        self.bc_z_r = DirichletBC( self.fsp.Q.sub( 0 ), z_r_const, self.rmsh.boundary_r )
        #mu_R = DirichletBC( self.fsp.Q.sub( 2 ), -0.05, self.rmsh.boundary_R )
        self.bc_z_R = DirichletBC( self.fsp.Q.sub( 0 ), z_R_const, self.rmsh.boundary_R )

        # all BCs
        self.bcs = [self.bc_z_r, self.bc_z_R]

    def change_nitsche_bc(self, omega_r, omega_R):
        self.F_N = self.alpha / self.rmsh.r_mesh * ((((self.bgeo.n_circle( self.fsp.omega ))[self.i] * self.fsp.omega[self.i] - omega_R) * ((self.bgeo.n_circle( self.fsp.omega ))[self.k] * self.geo.g( self.fsp.omega )[self.k, self.l] * self.fsp.nu_omega[self.l])) \
                                    * self.bgeo.sqrt_deth_circle( self.fsp.omega,self.rmsh.c_R ) * (1.0 / self.rmsh.R) * self.rmsh.ds_R \
                                    + (((self.bgeo.n_circle( self.fsp.omega ))[self.i] * self.fsp.omega[self.i] - omega_r) * ((self.bgeo.n_circle( self.fsp.omega ))[self.k] * self.geo.g( self.fsp.omega )[self.k, self.l] * self.fsp.nu_omega[self.l])) \
                                * self.bgeo.sqrt_deth_circle( self.fsp.omega,self.rmsh.c_r ) * (1.0 / self.rmsh.r) * self.rmsh.ds_r\
        )
        self.F = self.F_var + self.F_N
    
    # Define variational problem
    def define_variational_problem(self, kappa, z_r_const, z_R_const, omega_r, omega_R):
        self.F_z = (kappa * (self.geo.g_c( self.fsp.omega )[self.i, self.j] * (self.fsp.mu.dx(self.j)) * (self.fsp.nu_z.dx( self.i )) - 2.0 * self.fsp.mu * ((self.fsp.mu ** 2) - self.geo.K( self.fsp.omega )) * self.fsp.nu_z) + self.fsp.sigma * self.fsp.mu * self.fsp.nu_z) * self.geo.sqrt_detg(
            self.fsp.omega ) * self.rmsh.dx \
            - ( \
                        + (kappa * (self.bgeo.n_circle( self.fsp.omega ))[self.i] * self.fsp.nu_z * (self.fsp.mu.dx(self.i))) * self.bgeo.sqrt_deth_circle( self.fsp.omega, self.rmsh.c_r ) * (1.0 / self.rmsh.r) * self.rmsh.ds_r \
                        + (kappa * (self.bgeo.n_circle( self.fsp.omega ))[self.i] * self.fsp.nu_z * (self.fsp.mu.dx(self.i))) * self.bgeo.sqrt_deth_circle( self.fsp.omega, self.rmsh.c_R ) * (1.0 / self.rmsh.R) * self.rmsh.ds_R
            )

        self.F_omega = (- self.fsp.z * self.geo.Nabla_v( self.fsp.nu_omega, self.fsp.omega )[self.i, self.i] - self.fsp.omega[self.i] * self.fsp.nu_omega[self.i]) * self.geo.sqrt_detg( self.fsp.omega ) * self.rmsh.dx \
                + ((self.bgeo.n_circle( self.fsp.omega ))[self.i] * self.geo.g( self.fsp.omega )[self.i, self.j] * self.fsp.z * self.fsp.nu_omega[self.j]) * self.bgeo.sqrt_deth_circle( self.fsp.omega, self.rmsh.c_r ) * (1.0 / self.rmsh.r) * self.rmsh.ds_r \
                + ((self.bgeo.n_circle( self.fsp.omega ))[self.i] * self.geo.g( self.fsp.omega )[self.i, self.j] * self.fsp.z * self.fsp.nu_omega[self.j]) * self.bgeo.sqrt_deth_circle( self.fsp.omega, self.rmsh.c_R ) * (1.0 / self.rmsh.R) * self.rmsh.ds_R

        self.F_mu = ((self.geo.H( self.fsp.omega ) - self.fsp.mu) * self.fsp.nu_mu) * self.geo.sqrt_detg( self.fsp.omega ) * self.rmsh.dx

        

        '''self.F_N = self.alpha / self.rmsh.r_mesh * ((((self.bgeo.n_circle( self.fsp.omega ))[self.i] * self.fsp.omega[self.i] - omega_R) * ((self.bgeo.n_circle( self.fsp.omega ))[self.k] * self.geo.g( self.fsp.omega )[self.k, self.l] * self.fsp.nu_omega[self.l])) \
                                    * self.bgeo.sqrt_deth_circle( self.fsp.omega,self.rmsh.c_R ) * (1.0 / self.rmsh.R) * self.rmsh.ds_R \
                                    + (((self.bgeo.n_circle( self.fsp.omega ))[self.i] * self.fsp.omega[self.i] - omega_r) * ((self.bgeo.n_circle( self.fsp.omega ))[self.k] * self.geo.g( self.fsp.omega )[self.k, self.l] * self.fsp.nu_omega[self.l])) \
                                * self.bgeo.sqrt_deth_circle( self.fsp.omega,self.rmsh.c_r ) * (1.0 / self.rmsh.r) * self.rmsh.ds_r\
        ) '''
        # total functional for the mixed problem
        
        self.F_var = (self.F_z + self.F_omega + self.F_mu ) 
        self.change_nitsche_bc(omega_r, omega_R)

        #post-processing variational functional
        '''
        self.F_pp_nu = (self.fsp.nu[self.i] * self.fsp.nu_nu[self.i] + self.fsp.mu * self.geo.Nabla_v( self.fsp.nu_nu, self.fsp.omega )[self.i, self.i]) * self.geo.sqrt_detg( self.fsp.omega ) * self.rmsh.dx \
            - ((self.bgeo.n_circle( self.fsp.omega ))[self.i] * self.geo.g( self.fsp.omega )[self.i, self.j] * self.fsp.mu * self.fsp.nu_nu[self.j]) * self.bgeo.sqrt_deth_circle( self.fsp.omega, self.rmsh.c_r ) * (1.0 / self.rmsh.r) * self.rmsh.ds_r \
            - ((self.bgeo.n_circle( self.fsp.omega ))[self.i] * self.geo.g( self.fsp.omega )[self.i, self.j] * self.fsp.mu * self.fsp.nu_nu[self.j]) * self.bgeo.sqrt_deth_circle( self.fsp.omega, self.rmsh.c_r ) * (1.0 / self.rmsh.R) * self.rmsh.ds_R


        self.F_pp_tau = (self.fsp.nu[self.i] * self.geo.g_c( self.fsp.omega )[self.i, self.j] * (self.fsp.nu_tau.dx( self.j )) + self.fsp.tau * self.fsp.nu_tau) * self.geo.sqrt_detg( self.fsp.omega ) * self.rmsh.dx \
                - ((self.bgeo.n_circle( self.fsp.omega ))[self.i] * self.fsp.nu_tau * self.fsp.nu[self.i]) * self.bgeo.sqrt_deth_circle( self.fsp.omega, self.rmsh.c_r ) * (1.0 / self.rmsh.r) * self.rmsh.ds_r \
                - ((self.bgeo.n_circle( self.fsp.omega ))[self.i] * self.fsp.nu_tau * self.fsp.nu[self.i]) * self.bgeo.sqrt_deth_circle( self.fsp.omega, self.rmsh.c_R ) * (1.0 / self.rmsh.R) * self.rmsh.ds_R
        '''

