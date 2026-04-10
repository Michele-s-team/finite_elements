from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam 

Q = FunctionSpace(lmsh.mesh, 'DG', rpam.parameters['function_space_degree'])

# Define variational problem
u = Function(Q)
nu_u = TestFunction(Q)
f = Function(Q)

J_u = TrialFunction(Q)
u_exact = Function(Q)
