from fenics import *

import load_mesh as lmsh

function_space_degree = 1

# function space for u
P_u = VectorElement( 'P', triangle, function_space_degree )
P_v = VectorElement( 'P', triangle, function_space_degree )
element = MixedElement( [P_u, P_v] )
U = FunctionSpace(lmsh.mesh, element)
G = VectorFunctionSpace(lmsh.mesh, 'P', function_space_degree)


U_u_n = U.sub(0).collapse()
U_v_n = U.sub(1).collapse()

R = FunctionSpace(lmsh.mesh, 'P', function_space_degree)

J_psi = TrialFunction(U)
psi = Function(U)
nu_u_n, nu_v_n = TestFunctions( U )

#fields at the preceeding steps
u_n_1 = Function(U_u_n)
u_n = Function(U_u_n)
v_n_1 = Function(U_v_n)
v_n = Function(U_v_n)

# other fields
u_l = Function(U_u_n)
rho = Function(R)
g = Function(G)

