from fenics import *
from mshr import *

#import boundary_geometry as bgeo
#import read_mesh_ring as rmsh
# import read_mesh_square_no_circle as rmsh


# Define function spaces
#finite elements for sigma .... omega
'''
z
omega_i = \partial_i z
mu = H(omega)
tau = Nabla_i nu^i 
'''
class fsp:
    def __init__(self, bgeo):
        degree_function_space = 1
        P_z = FiniteElement( 'P', triangle, degree_function_space )
        P_omega = VectorElement( 'P', triangle, degree_function_space )
        P_mu = FiniteElement( 'P', triangle, degree_function_space )


        element = MixedElement( [P_z, P_omega, P_mu] )
        #total function space
        self.Q = FunctionSpace(bgeo.mesh, element)
        #function spaces for z, omega, eta and theta
        self.Q_z = self.Q.sub( 0 ).collapse()
        self.Q_omega = self.Q.sub( 1 ).collapse()
        self.Q_mu = self.Q.sub( 2 ).collapse()

        self.Q_sigma = FunctionSpace( bgeo.mesh, 'P', 1 )
        #the function spaces for nu and tau are for post-processing only
        self.Q_nu = VectorFunctionSpace( bgeo.mesh, 'CG', degree_function_space )
        self.Q_tau = FunctionSpace( bgeo.mesh, 'P', degree_function_space )
        self.Q_F = FunctionSpace( bgeo.mesh, "P", degree_function_space)
        self.Q_e = TensorFunctionSpace( bgeo.mesh, 'CG', degree_function_space, (2,3))

        # Define functions
        self.J_psi = TrialFunction( self.Q )
        self.psi = Function( self.Q )
        self.nu_z, self.nu_omega, self.nu_mu = TestFunctions( self.Q )

        self.J_pp_nu = TrialFunction( self.Q_nu )
        self.J_pp_tau = TrialFunction( self.Q_tau )
        self.nu_nu = TestFunction(self.Q_nu)
        self.nu_tau = TestFunction(self.Q_tau)


        #these functions are used to print the solution to file
        self.sigma = Function(self.Q_sigma)
        self.nu = Function(self.Q_nu)
        self.tau = Function(self.Q_tau)

        self.z_output = Function(self.Q_z)
        self.omega_output = Function(self.Q_omega)
        self.mu_output = Function( self.Q_mu )

        self.z_exact = Function( self.Q_z )
        self.omega_exact = Function( self.Q_omega )
        self.mu_exact = Function( self.Q_mu )



        self.nu_exact = Function( self.Q_nu )
        self.tau_exact = Function( self.Q_tau )

        # omega_0, z_0 are used to store the initial conditions
        self.z_0 = Function( self.Q_z )
        self.omega_0 = Function( self.Q_omega )
        self.mu_0 = Function( self.Q_mu )
        self.nu_0 = Function( self.Q_nu )
        self.tau_0 = Function( self.Q_tau )

        self.z, self.omega, self.mu = split( self.psi )
        self.assigner = FunctionAssigner( self.Q, [self.Q_z, self.Q_omega, self.Q_mu] )