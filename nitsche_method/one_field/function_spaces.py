from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam

Q = FunctionSpace(lmsh.mesh, 'P', rpam.parameters['function_space_degree'])

# Define variational problem
u = Function(Q)
u_D = Function(Q)
nu_u = TestFunction(Q)
f = Function(Q)

J_u = TrialFunction(Q)
