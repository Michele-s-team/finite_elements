from fenics import *
from mshr import *

import boundary_geometry as bgeo

function_space_degree = 4


Q = FunctionSpace( mesh, 'P', function_space_degree )
V = VectorFunctionSpace( mesh, 'P', function_space_degree )
T = TensorFunctionSpace( mesh, 'P', function_space_degree, shape=(2, 2) )


# Define variational problem
u = Function( Q )
nu_u = TestFunction( Q )
f = Function( Q )
grad_u = Function( V )
J_u = TrialFunction( Q )
u_exact = Function( Q )

# Define post-processing (pp) variational problem
# hess_u is a tensor which is the Hessian matrix of u: hess_u[i, j] = \partial_i \partial_j u
hess_u = Function( T )
nu_hess_u = TestFunction( T )
hess_u_exact = Function( T )
J_hess_u = TrialFunction( T )