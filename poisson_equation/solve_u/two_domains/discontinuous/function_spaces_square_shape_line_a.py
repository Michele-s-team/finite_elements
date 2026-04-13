'''
here 'a' anb 'b' refer to the two domains
    - 'a' is the label for one domain
    - 'b' is the label for the other domain
'''
from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam 

Q = FunctionSpace(lmsh.mesh[0], 'DG', rpam.parameters['function_space_degree'])

# Define variational problem
u = Function(Q)
nu_u = TestFunction(Q)

f_a = Function(Q)
f_b = Function(Q)

d = Function(Q)

J_u = TrialFunction(Q)

u_exact_a = Function(Q)
u_exact_b = Function(Q)
