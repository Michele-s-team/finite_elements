from fenics import *
import importlib

import mesh.load as lmsh
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

# This enforces periodic boundary conditions which map the l vertex into the r vertex or mesh 1
class PeriodicBoundary(SubDomain):
    # Identify the "target domain": the left vertex
    def inside(self, x, on_boundary):
        return near(x[0], lmsh.mesh_parameters[1]['x_l']) and on_boundary

    # Map the other boundaries to the "target domain"
    def map(self, x, y):
        if near(x[0], lmsh.mesh_parameters[1]['x_r']):
            # right vertex → left vertex
            y[0] = lmsh.mesh_parameters[1]['x_l']
        else:
            # Required: set unmapped points to identity
            y[0] = x[0]
            

periodic_boundary = PeriodicBoundary()


# 1. function spaces
Q_U = VectorFunctionSpace(lmsh.mesh[1], 'P', rpam.parameters['function_space_degree'], dim=2, constrained_domain=periodic_boundary)
Q_psi0 = FunctionSpace(lmsh.mesh[1], 'P', rpam.parameters['function_space_degree'])

P_nu = FiniteElement('P', interval, rpam.parameters['function_space_degree'])
P_dpsi = FiniteElement('P', interval, rpam.parameters['function_space_degree'])
element = MixedElement( [P_nu, P_dpsi] )

Q_nu_and_dpsi = FunctionSpace(lmsh.mesh[1], element, constrained_domain=periodic_boundary)

Q_nu= Q_nu_and_dpsi.sub(0).collapse()
Q_dpsi= Q_nu_and_dpsi.sub(1).collapse()


Q_mu = FunctionSpace(lmsh.mesh[1], 'P', rpam.parameters['function_space_degree'], constrained_domain=periodic_boundary)



# 2 fields
U = Function(Q_U)

nu_and_dpsi = Function(Q_nu_and_dpsi)
nu, dpsi = split( nu_and_dpsi )

mu = Function(Q_mu)

#  3 test functions
nu_U = TestFunction(Q_U)
nu_nu, nu_dpsi = TestFunctions( Q_nu_and_dpsi )
nu_mu = TestFunction(Q_mu)


# 4 jacobians
J_U = TrialFunction(Q_U)
J_nu_and_dpsi = TrialFunction(Q_nu_and_dpsi)
J_mu = TrialFunction(Q_mu)

psi0 = Function(Q_psi0)
ys = Function(Q_U)



