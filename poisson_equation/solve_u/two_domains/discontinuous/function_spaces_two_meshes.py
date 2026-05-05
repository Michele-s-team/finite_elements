'''
this module defines function spaces for problem involving two meshes
'''

from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam 

Q = FunctionSpace(lmsh.mesh[0], 'DG', rpam.parameters['function_space_degree'])

# Define variational problem
u = Function(Q)
nu_u = TestFunction(Q)

f = Function(Q)
d = Function(Q)
e = Function(Q)

J_u = TrialFunction(Q)

u_exact = Function(Q)
