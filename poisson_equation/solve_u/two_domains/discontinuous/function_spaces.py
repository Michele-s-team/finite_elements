from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam 

Q = FunctionSpace(lmsh.mesh[0], 'DG', rpam.parameters['function_space_degree'])

# Define variational problem
u = Function(Q)
nu_u = TestFunction(Q)

f_shape = Function(Q)
f_square = Function(Q)

d = Function(Q)

J_u = TrialFunction(Q)

u_exact_shape = Function(Q)
u_exact_square = Function(Q)
