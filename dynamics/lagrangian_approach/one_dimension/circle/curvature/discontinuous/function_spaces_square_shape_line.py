from fenics import *
import importlib

import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
the variables for the problem are
    - 'mu': the curvature of the curve x_s in the current configuration
    - 'u': displacement field of \partial \Omega_circle, defined on the entire domain (it can represent displacement of points also outside \partial \Omega^y)
    - 'grad_u': grad_u[i, j] = \partial u_i / \partial y_j
    - 'f': tangent vector to the curve y_s in the reference configuration, extended to the whole domain
    - 'e': tangent vector to the curve x_s in the current configuration, extended to the whole domain
    - 'n': unit normal to x_t pointing outwards \Omega_circle

'''

#1 define elements 

D_mu = FiniteElement('DG', triangle, rpam.parameters['function_space_degree'])
D_grad_u = TensorElement('DG', triangle, rpam.parameters['function_space_degree'])

element = MixedElement([D_mu, D_grad_u])

Q = FunctionSpace(lmsh.mesh[0], element)


Q_mu = Q.sub(0).collapse()
Q_grad_u = Q.sub(1).collapse()

V = VectorFunctionSpace(lmsh.mesh[0], 'DG', rpam.parameters['function_space_degree'])

# fields
psi = Function(Q)
mu, grad_u = split(psi)


f = Function(V)
nu = Function(V)
b = Function(Q_grad_u)

u = Function(V)


# test functions
nu_mu, nu_grad_u = TestFunctions(Q)


# jacobian
J_psi = TrialFunction(Q)





