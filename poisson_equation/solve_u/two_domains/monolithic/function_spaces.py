from fenics import *
import importlib

import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)


# mesh i has no sub-meshes 

Q = FunctionSpace(lmsh.mesh[0], 'P', rpam.parameters['function_space_degree'])

# Define variational problem
u = Function(Q)
nu_u = TestFunction(Q)
f = Function(Q)

J_u = TrialFunction(Q)
u_exact = Function(Q)





