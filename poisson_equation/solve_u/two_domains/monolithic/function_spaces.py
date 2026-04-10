from fenics import *
import importlib

import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)


Q = FunctionSpace(lmsh.mesh[0], 'P', rpam.parameters['function_space_degree'])
V = VectorFunctionSpace(lmsh.mesh[0], 'P', rpam.parameters['function_space_degree'])

u = Function(Q)

f_shape = Function(Q)
f_square = Function(Q)
g_shape = Function(Q)

nu_u = TestFunction(Q)

J_u = TrialFunction(Q)

u_exact_shape = Function(Q)
u_exact_square = Function(Q)

grad_u_exact_shape = Function(V)
grad_u_exact_square = Function(V)





