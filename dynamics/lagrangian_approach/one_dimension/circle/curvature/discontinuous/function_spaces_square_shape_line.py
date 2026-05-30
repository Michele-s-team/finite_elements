from fenics import *
import importlib

import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
the variables for the problem are
    - 'u': deformation field of the shape and mesh
'''

#1 define elements 

#1.1 fluid 

#1.2 elastic body and mesh
Q = FunctionSpace(lmsh.mesh[0], 'DG', rpam.parameters['function_space_degree'])
V = VectorFunctionSpace(lmsh.mesh[0], 'DG', rpam.parameters['function_space_degree'])

# fields
mu = Function(Q)

n_0 = Function(V)
t_0 = Function(V)

n = Function(V)
t = Function(V)


# test functions
nu_mu = TestFunction(Q)

# jacobian
J = TrialFunction(Q)





