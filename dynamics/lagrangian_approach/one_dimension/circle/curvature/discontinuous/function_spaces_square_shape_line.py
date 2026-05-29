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
D_u = VectorElement('DG', triangle, rpam.parameters['u_function_space_degree'])

element = MixedElement([D_u])



#2 define function spaces

#2.1 global function space
Q = FunctionSpace(lmsh.mesh[0], element)

#2.2 collapsed function spaces

Q_u = Q.sub(0).collapse()


#3 define fields

# 3.1 psi contains all fields
psi = Function(Q)
u = split(psi)


# 3.2 auxiliary fields










# 3.3 test functions
nu_u = TestFunctions(Q)

# 3.4 jacobian
J_psi = TrialFunction(Q)

# 3.5 function assigner

assigner = FunctionAssigner(Q, [Q_u])




